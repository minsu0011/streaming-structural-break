"""Isolated paired timing of tuple and resident calls to the identical kernel."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
import json,sys,time
from pathlib import Path
import numpy as np
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.models.cross_order import CrossOrderBlendBundle
from src.streaming.fused_cross_order import PreparedFusedCrossOrderPredictor,implementation_hash
from src.data.loader import load_training
from src.validation.splits import FrozenSplit
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    names={p.name for p in (ROOT/'scripts').glob('*.py')};ancestors={p.pid for p in psutil.Process().parents()};active=[]
    for process in psutil.process_iter(['pid','cmdline']):
        try:
            command=process.info['cmdline'] or []
            if process.pid!=os.getpid() and process.pid not in ancestors and (any(Path(v).name in names for v in command) or 'pytest' in command):active.append(process.pid)
        except (psutil.AccessDenied,psutil.NoSuchProcess):pass
    if active:raise RuntimeError('Another project worker is active; isolated paired timing refused')
    output=ROOT/'artifacts/real/latency/resident_state_dispatch_comparison.json'
    if output.exists():raise RuntimeError('Paired comparison already exists')
    eid='M7_cross_order_AR4_LGB_AR8_HGB_equal';path=ROOT/'artifacts/models/real'/eid/'fold_0.joblib'
    split=FrozenSplit(ROOT/'artifacts/validation');series=next(load_training(ROOT/'data/raw',split=split));h=series.historical.astype(np.float32)
    points=np.tile(series.online.astype(np.float32),100000//len(series.online)+1)[:100000]
    prepared=PreparedFusedCrossOrderPredictor(CrossOrderBlendBundle.load(path),fast_initialization=True)
    a=prepared.new_state(h);b=prepared.new_state(h)
    expected=np.array([a.predict_one_tuple(p) for p in points],dtype=np.float32)
    actual=np.array([b.predict_one(p) for p in points],dtype=np.float32);np.testing.assert_array_equal(actual,expected)
    rows=[]
    for repetition in range(9):
        for name in (('tuple','resident') if repetition%2==0 else ('resident','tuple')):
            state=prepared.new_state(h);method=state.predict_one_tuple if name=='tuple' else state.predict_one
            started=time.perf_counter_ns()
            for point in points:method(point)
            rows.append({'repetition':repetition,'mode':name,'us_per_point':(time.perf_counter_ns()-started)/len(points)/1000})
    ratios=[next(r['us_per_point'] for r in rows if r['repetition']==i and r['mode']=='tuple')/next(r['us_per_point'] for r in rows if r['repetition']==i and r['mode']=='resident') for i in range(9)]
    report={'status':'PASS','timestamp':utc_now(),'candidate':eid,'model_sha256':sha256(path),'cross_order_fused_source_hash':implementation_hash(),
        'policy':'Nine alternating paired 100000-point runs; same real DEV value buffer. Point calls only, historical initialization outside timing.',
        'different_float32_rows':0,'median_speed_ratio':float(np.median(ratios)),'paired_tuple_over_resident_ratios':ratios,'timings':rows,
        'isolated_from_project_workers':True,'other_project_worker_pids':active,'script_sha256':sha256(__file__),
        'scope':'Windows CPU local predictor, no socket/cloud claim','LOCAL_FINAL_SEAL_OPENED':False,'reduced_targets_used':False}
    write_json(output,report);print('RESIDENT PAIRED PASS',report['median_speed_ratio'],ratios,flush=True)


if __name__=='__main__':main()
