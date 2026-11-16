"""Matched scalar M7/BL80 benchmark after queue quiescence; no model fitting."""
from pathlib import Path
import sys
import argparse
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now

def main():
    import numpy as np
    import time
    import pickle
    import psutil
    from src.next9.runtime_loading_v2 import load_heads,load_baseline
    from src.next9.streaming import IntegratedStream
    from src.next9.negative_controls_v2 import BACKGROUNDS,SEEDS,H_LENGTHS,generate
    from src.next9.queue import identity_alive
    # A controller may remain alive, but no owned worker may compete with this
    # benchmark. Do not silently publish timings measured during model fitting.
    for p in (BASE/'runtime/queue').glob('*/attempt_*/START.json'):
        if (p.parent/'DONE.json').exists():continue
        r=read_json(p)
        assert not identity_alive(r['pid'],r['process_created']),'Wait for owned queue worker completion'
    metadata=dict(timestamp=utc_now(),CPU_count=psutil.cpu_count(),RAM_percent=psutil.virtual_memory().percent,
        benchmark_process=psutil.Process().pid,other_system_workloads_not_controlled=True)
    records=[]
    for fold in range(5):
        for mode in ['NORMAL','EARLY']:
            with activity(f'Matched scalar latency outer{fold} {mode},24fixed synthetic paths','RUNTIME_BENCHMARK','B9_NORMALIZATION_STATE_AVAILABILITY'):
                heads=load_heads(fold,mode);baseline=load_baseline(fold,mode)
                warm_h,warm_o,_=generate('GAUSSIAN','NONE',SEEDS[0],512)
                b=baseline.new_state(warm_h);m=IntegratedStream(heads,baseline,warm_h)
                for value in warm_o[:32]:b.predict_one(float(value));m.predict_one(float(value))
                durations={'BL80':[],'M7':[]};initial={'BL80':[],'M7':[]};sizes=[]
                cases=[(background,seed,h_length) for background in BACKGROUNDS for seed in SEEDS for h_length in H_LENGTHS]
                for k,(background,seed,h_length) in enumerate(cases):
                    h,o,_=generate(background,'NONE',seed,h_length)
                    objects={}
                    for name in (['BL80','M7'] if k%2==0 else ['M7','BL80']):
                        start=time.perf_counter_ns()
                        objects[name]=baseline.new_state(h) if name=='BL80' else IntegratedStream(heads,baseline,h)
                        initial[name].append(time.perf_counter_ns()-start)
                    for t,point in enumerate(o):
                        predictions={}
                        for name in (['BL80','M7'] if (k+t)%2==0 else ['M7','BL80']):
                            start=time.perf_counter_ns();predictions[name]=objects[name].predict_one(float(point))
                            durations[name].append(time.perf_counter_ns()-start)
                        if heads.correction is None:assert predictions['BL80']==predictions['M7']
                    sizes.append(dict(BL80_numeric_state_bytes=len(pickle.dumps(objects['BL80'].args,protocol=5)),
                        M7_snapshot_with_fitted_references_bytes=len(pickle.dumps(objects['M7'].snapshot(),protocol=5))))
                def summary(values):
                    return {key:float(np.quantile(values,q)/1e3) for key,q in [('median_us',.5),('p95_us',.95),('p99_us',.99)]}
                latency={k:summary(v) for k,v in durations.items()};init={k:summary(v) for k,v in initial.items()}
                median_ratio=latency['M7']['median_us']/latency['BL80']['median_us']
                p99_ratio=latency['M7']['p99_us']/latency['BL80']['p99_us']
                records.append(dict(fold=fold,mode=mode,cases=len(cases),points=len(durations['M7']),latency=latency,initialization=init,
                    median_ratio=median_ratio,p99_ratio=p99_ratio,passed=median_ratio<=3 and p99_ratio<=5,
                    median_state_sizes={key:float(np.median([r[key] for r in sizes])) for key in sizes[0]},
                    M7_exact_BL_bypass=heads.correction is None,alternating_per_point_execution_order=True,warmup_excluded=True))
    atomic(BASE/'LATENCY_NEXT9.json',dict(status='PASS' if all(r['passed'] for r in records) else 'FAIL',
        **metadata,records=records,thresholds=dict(median_ratio_max=3,p99_ratio_max=5),
        interpretation='Matched excluded-fold BL80; each of10 fixed procedures must pass. Init and serialized sizes descriptive, no thresholds.',
        no_canonical_role_change=True,source_sha256=digest(Path(__file__))))

if __name__=='__main__':main()
