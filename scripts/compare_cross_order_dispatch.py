"""Alternating paired timing of the exact original and single-call mean paths."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
import json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.models.cross_order import CrossOrderBlendBundle
from src.streaming.fused_cross_order import PreparedFusedCrossOrderPredictor,implementation_hash
from src.data.loader import load_training
from src.validation.splits import FrozenSplit
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    eid='M7_cross_order_AR4_LGB_AR8_HGB_equal';path=ROOT/'artifacts/models/real'/eid/'fold_0.joblib'
    split=FrozenSplit(ROOT/'artifacts/validation');series=next(load_training(ROOT/'data/raw',split=split));h=series.historical.astype(np.float32)
    points=np.tile(series.online.astype(np.float32),100000//len(series.online)+1)[:100000]
    prepared=PreparedFusedCrossOrderPredictor(CrossOrderBlendBundle.load(path),fast_initialization=True)
    a=prepared.new_state(h);b=prepared.new_state(h)
    expected=np.array([a.predict_one_reference(p) for p in points],dtype=np.float32)
    actual=np.array([b.predict_one(p) for p in points],dtype=np.float32);np.testing.assert_array_equal(actual,expected)
    rows=[]
    for repetition in range(9):
        for name in (('reference','single_call') if repetition%2==0 else ('single_call','reference')):
            detector=prepared.new_state(h);method=detector.predict_one_reference if name=='reference' else detector.predict_one
            started=time.perf_counter_ns()
            for point in points:method(point)
            rows.append({'repetition':repetition,'mode':name,'us_per_point':(time.perf_counter_ns()-started)/len(points)/1000})
    pairs=[next(r['us_per_point'] for r in rows if r['repetition']==i and r['mode']=='reference')/next(r['us_per_point'] for r in rows if r['repetition']==i and r['mode']=='single_call') for i in range(9)]
    report={'status':'PASS','timestamp':utc_now(),'candidate':eid,'model_sha256':sha256(path),'cross_order_fused_source_hash':implementation_hash(),
        'policy':'nine alternating paired repetitions, 100000 identical real DEV values per repeat; initialization outside timing; no model score',
        'different_float32_rows':int(np.count_nonzero(actual!=expected)),'paired_reference_over_single_call_ratios':pairs,'median_speed_ratio':float(np.median(pairs)),
        'point_timings':rows,'scope':'Single local process, sequential before new training workers. Timing variability is retained; no claim about cloud IPC.',
        'LOCAL_FINAL_SEAL_OPENED':False,'reduced_targets_used':False}
    write_json(ROOT/'artifacts/real/latency/cross_order_dispatch_comparison.json',report)
    print('PAIRED DISPATCH',report['median_speed_ratio'],pairs,flush=True)


if __name__=='__main__':main()
