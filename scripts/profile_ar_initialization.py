"""Attribute the original full-column MAD initialization cost on DEV histories."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse, json, sys, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.models.representations import bundle_class, ar_input_order
from src.features.ar_residual_input import historical_filter
from src.features.ar_order_input import historical_filter as order_filter
from src.features.streaming import StreamingFeatureState
from src.features.historical_calibration import _historical_replay
from src.data.loader import load_training
from src.validation.splits import FrozenSplit
from src.utils.artifacts import write_json,utc_now


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',required=True);args=parser.parse_args()
    state=json.loads((ROOT/'artifacts/real/RESEARCH_STATE.json').read_text());result=next(r for r in state['results'] if r['experiment_id']==args.candidate)
    representation=result['candidate']['representation'];model=bundle_class(representation).load(ROOT/'artifacts/models/real'/args.candidate/'fold_0.joblib')
    if model.policy.method!='historical_median_mad':raise ValueError('Profile follows the original full-column MAD reference path')
    split=FrozenSplit(ROOT/'artifacts/validation');rows=[]
    def stages(h):
        start=time.perf_counter_ns();order=ar_input_order(representation)
        _,_,_,residual=historical_filter(h,model.feature_config) if order is None else order_filter(h,model.feature_config,order)
        a=time.perf_counter_ns();s=StreamingFeatureState(residual,config=model.feature_config)
        b=time.perf_counter_ns();reference=_historical_replay(residual,s.reference,s.baseline,s.thresholds,s.ring.copy(),s.ew.copy(),s.counters.copy(),s.output.copy(),s.parameters,s.alphas,s.lags)
        c=time.perf_counter_ns();burn=min(model.policy.burn_in,len(reference)//2);columns=np.ascontiguousarray(reference[burn:].T,dtype=np.float64)
        d=time.perf_counter_ns();center=np.median(columns,axis=1);scale=np.maximum(np.median(abs(columns-center[:,None]),axis=1)*1.4826,model.policy.standard_deviation_floor)
        e=time.perf_counter_ns()
        return {'ar_filter_ms':(a-start)/1e6,'inner_reference_ms':(b-a)/1e6,'historical_feature_replay_ms':(c-b)/1e6,'transpose_copy_ms':(d-c)/1e6,'median_mad_ms':(e-d)/1e6,'total_ms':(e-start)/1e6}
    for series in load_training(ROOT/'data/raw',split=split):
        h=series.historical.astype(np.float32)
        if not rows:stages(h)
        rows.append({'dataset_id':series.dataset_id,'historical_length':len(h),**stages(h)})
        if len(rows)==128:break
    keys=[k for k in rows[0] if k.endswith('_ms')]
    summary={key:{'mean':float(np.mean([r[key] for r in rows])),'median':float(np.median([r[key] for r in rows]))} for key in keys}
    report={'status':'PASS','timestamp':utc_now(),'candidate':args.candidate,'series':len(rows),'selection':'first 128 ascending DEV IDs; no score-based probe choice',
        'summary':summary,'per_series':rows,'scope':'Original full-column MAD stage attribution under concurrent research load, before selected-column calibration. Final isolated production timing remains authoritative.',
        'LOCAL_FINAL_SEAL_OPENED':False,'reduced_targets_used':False}
    write_json(ROOT/'artifacts/real/latency/initialization_profile'/f'{args.candidate}.json',report);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
