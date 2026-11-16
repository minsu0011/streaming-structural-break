"""Locate frozen M7 inference costs; this is not a latency certification run."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now
from src.next9.boundary import install_workspace_boundary

def main():
    violations=install_workspace_boundary()
    import cProfile
    import pstats
    import time
    from src.next9.runtime_loading_v3 import load_heads,load_baseline
    from src.next9.streaming import IntegratedStream
    from src.next9.negative_controls_v2 import generate,SEEDS,H_LENGTHS
    heads=load_heads(0,'NORMAL');baseline=load_baseline(0,'NORMAL')
    h,o,_=generate('GAUSSIAN','NONE',SEEDS[0],512)
    warm=IntegratedStream(heads,baseline,h)
    for x in o[:32]:warm.predict_one(float(x))
    records=[]
    with activity('Profile frozen outer0 NORMAL dedicated M7 and BL on six fixed Gaussian paths','RUNTIME_PROFILING','B9_NORMALIZATION_STATE_AVAILABILITY'):
        for name in ['BL80','M7']:
            profile=cProfile.Profile();cpu=0.;wall=0.;count=0
            for seed in SEEDS:
                for length in H_LENGTHS:
                    h,o,_=generate('GAUSSIAN','NONE',seed,length)
                    state=baseline.new_state(h) if name=='BL80' else IntegratedStream(heads,baseline,h)
                    start_cpu=time.process_time();start=time.perf_counter();profile.enable()
                    for point in o:state.predict_one(float(point))
                    profile.disable();cpu+=time.process_time()-start_cpu;wall+=time.perf_counter()-start;count+=len(o)
            stats=pstats.Stats(profile);functions=[]
            for (filename,line,function),(primitive,calls,total,cumulative,callers) in stats.stats.items():
                functions.append(dict(file=filename,line=line,function=function,primitive_calls=primitive,calls=calls,total_seconds=total,cumulative_seconds=cumulative))
            functions.sort(key=lambda r:r['cumulative_seconds'],reverse=True)
            records.append(dict(method=name,points=count,profiled_process_CPU_us_per_point=cpu/count*1e6,
                profiled_wall_us_per_point=wall/count*1e6,top_functions=functions[:60]))
        assert not violations
        atomic(BASE/'M7_RUNTIME_PROFILE_NEXT9.json',dict(status='COMPLETE',timestamp=utc_now(),fold=0,mode='NORMAL',records=records,
            M7_qualified=heads.correction is not None,rule_count=len(heads.spec['rule_specs']),
            profiling_overhead_included=True,other_owned_worker_may_run=True,not_runtime_gate_measurement=True,
            fixed_synthetic_cases=True,no_quality_labels_or_model_fits=True))
        print([{k:v for k,v in r.items() if k!='top_functions'} for r in records],flush=True)

if __name__=='__main__':main()
