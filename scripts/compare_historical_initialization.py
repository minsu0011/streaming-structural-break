"""Alternating-order paired initialization timings on fixed ascending DEV IDs."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[key]='1'
import argparse, json, sys, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.models.calibrated import CalibratedBundle
from src.streaming.fused import PreparedFusedPredictor
from src.streaming.fast_initialization import PreparedFastInitFusedPredictor,implementation_hash
from src.data.loader import load_training
from src.validation.splits import FrozenSplit
from src.utils.artifacts import write_json,utc_now,sha256


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',required=True);parser.add_argument('--series',type=int,default=128);args=parser.parse_args()
    registry=json.loads((ROOT/'artifacts/real/RESEARCH_STATE.json').read_text());result=next(r for r in registry['results'] if r['experiment_id']==args.candidate)
    from src.models.representations import bundle_class,ar_prefix
    representation=result['candidate'].get('representation','base')
    model_path=ROOT/'artifacts/models/real'/args.candidate/'fold_0.joblib';model=bundle_class(representation).load(model_path)
    if ar_prefix(representation):
        from src.streaming.fused_ar import PreparedFusedARPredictor
        original=PreparedFusedARPredictor(model,fast_initialization=False);optimized=PreparedFusedARPredictor(model,fast_initialization=True)
    else:original=PreparedFusedPredictor(model);optimized=PreparedFastInitFusedPredictor(model)
    split=FrozenSplit(ROOT/'artifacts/validation');timings={'original':[],'optimized':[]};rows=[]
    for count,series in enumerate(load_training(ROOT/'data/raw',split=split)):
        if count>=args.series:break
        h=series.historical.astype(np.float32)
        if count==0:
            original.new_state(h).predict_one(0.);optimized.new_state(h).predict_one(0.)
        states={}
        for label,prepared in ([('original',original),('optimized',optimized)] if count%2==0 else [('optimized',optimized),('original',original)]):
            start=time.perf_counter_ns();states[label]=prepared.new_state(h);timings[label].append((time.perf_counter_ns()-start)/1e9)
        for a,b in zip(states['original'].state_args,states['optimized'].state_args):np.testing.assert_array_equal(a,b)
        if hasattr(states['original'],'ar_args'):
            for a,b in zip(states['original'].ar_args,states['optimized'].ar_args):np.testing.assert_array_equal(a,b)
        left=states['original'].calibration_args;right=states['optimized'].calibration_args;active=right[2]
        tree_used=np.zeros(len(active),dtype=bool);tree_used[np.unique(optimized.tree_args[1][optimized.tree_args[1]>=0])]=True
        np.testing.assert_array_equal(active,left[2]&tree_used)
        for a,b in zip(left[:2],right[:2]):np.testing.assert_array_equal(a[active],b[active])
        for a,b in zip(left[3:],right[3:]):np.testing.assert_array_equal(a,b)
        for point in series.online[:min(32,len(series.online))]:
            np.testing.assert_array_equal(np.float32(states['original'].predict_one(point)),np.float32(states['optimized'].predict_one(point)))
        rows.append({'dataset_id':series.dataset_id,'historical_length':len(h),'original_seconds':timings['original'][-1],'optimized_seconds':timings['optimized'][-1]})
    report={'status':'PASS','timestamp':utc_now(),'candidate':args.candidate,'model_sha256':sha256(model_path),'series':len(rows),
        'all_initial_state_arrays_bitwise_equal':True,'used_calibration_values_bitwise_equal':True,'unused_calibration_slots':'zero center/unit scale/unmasked; inaccessible to every fitted tree split','prefix_predictions_bitwise_equal':True,
        'timing_order':'alternate original/optimized order by ascending development-series position','shared_cpu_load':True,
        'original_mean_seconds':float(np.mean(timings['original'])),'optimized_mean_seconds':float(np.mean(timings['optimized'])),
        'original_median_seconds':float(np.median(timings['original'])),'optimized_median_seconds':float(np.median(timings['optimized'])),
        'total_time_speedup':float(np.sum(timings['original'])/np.sum(timings['optimized'])),'series_timings':rows,
        'initialization_source_hash':implementation_hash(),'split_sha256':split.digest,'LOCAL_FINAL_SEAL_OPENED':False,'reduced_targets_used':False}
    path=ROOT/'artifacts/real/initialization'/f'{args.candidate}_selected_columns.json';write_json(path,report)
    print('INITIALIZATION PASS',args.candidate,'series',len(rows),'speedup',report['total_time_speedup'],flush=True)


if __name__=='__main__':main()
