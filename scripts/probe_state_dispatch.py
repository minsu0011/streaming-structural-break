"""Bounded engineering probe: pass a resident typed state instead of array tuples.

This prototype does not change the production dispatcher or saved predictors.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
import json,sys,time
from pathlib import Path
import numpy as np
from numba import typeof
from numba.experimental import jitclass
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.models.cross_order import CrossOrderBlendBundle
from src.streaming.fused_cross_order import PreparedFusedCrossOrderPredictor,fused_equal_average,implementation_hash
from src.data.loader import load_training
from src.validation.splits import FrozenSplit
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    output=ROOT/'artifacts/real/latency/resident_state_prototype.json'
    if output.exists():raise RuntimeError('Historical prototype result already exists; use compare_resident_dispatch.py for the integrated path')
    eid='M7_cross_order_AR4_LGB_AR8_HGB_equal';path=ROOT/'artifacts/models/real'/eid/'fold_0.joblib'
    split=FrozenSplit(ROOT/'artifacts/validation');series=next(load_training(ROOT/'data/raw',split=split));h=series.historical.astype(np.float32)
    points=np.tile(series.online.astype(np.float32),100000//len(series.online)+1)[:100000]
    prepared=PreparedFusedCrossOrderPredictor(CrossOrderBlendBundle.load(path),fast_initialization=True)
    sample=prepared.new_state(h)
    @jitclass([('args',typeof(sample.call_args))])
    class Resident:
        def __init__(self,args):self.args=args
        def predict_one(self,point):return fused_equal_average(point,self.args[0],self.args[1])
    a=prepared.new_state(h);b=prepared.new_state(h);resident=Resident(b.call_args)
    start=time.perf_counter();resident.predict_one(float(points[0]));cold=time.perf_counter()-start
    b=prepared.new_state(h);resident=Resident(b.call_args)
    expected=np.array([a.predict_one(p) for p in points],dtype=np.float32)
    actual=np.array([resident.predict_one(float(p)) for p in points],dtype=np.float32);np.testing.assert_array_equal(actual,expected)
    rows=[]
    for repetition in range(7):
        for name in (('tuple_call','resident') if repetition%2==0 else ('resident','tuple_call')):
            state=prepared.new_state(h)
            obj=Resident(state.call_args) if name=='resident' else state
            method=obj.predict_one
            started=time.perf_counter_ns()
            for point in points:float(method(float(point)))
            rows.append({'repetition':repetition,'mode':name,'us_per_point':(time.perf_counter_ns()-started)/len(points)/1000})
    pairs=[next(r['us_per_point'] for r in rows if r['repetition']==i and r['mode']=='tuple_call')/next(r['us_per_point'] for r in rows if r['repetition']==i and r['mode']=='resident') for i in range(7)]
    report={'status':'PASS','timestamp':utc_now(),'candidate':eid,'model_sha256':sha256(path),'fused_source_hash':implementation_hash(),
        'prototype_source_sha256':sha256(__file__),'production_changed':False,'scope':'100000 repeated real DEV values; seven alternating paired timing runs. Other diagnostics may share CPU; confirm isolated before any promotion.',
        'cold_resident_compilation_seconds':cold,'different_float32_rows':0,'median_speed_ratio':float(np.median(pairs)),'paired_ratios':pairs,'timings':rows,
        'documentation':'https://numba.readthedocs.io/en/stable/user/jitclass.html','LOCAL_FINAL_SEAL_OPENED':False,'reduced_targets_used':False}
    write_json(ROOT/'artifacts/real/latency/resident_state_prototype.json',report);print('RESIDENT PROBE',report['median_speed_ratio'],pairs,flush=True)


if __name__=='__main__':main()
