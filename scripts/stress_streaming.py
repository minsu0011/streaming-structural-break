"""Long-stream numerical, speed and bounded-memory engineering audit."""
import sys
import time
from pathlib import Path
import numpy as np
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.features.streaming import StreamingFeatureState,_update,replay
from src.models.baselines import BaselineDetector
from src.utils.artifacts import write_json

def deep_size(obj,seen=None):
    """Reachable Python size, including containers and retained array owners.

    Shared objects count once. NumPy owners include their data in getsizeof;
    views retain the owner, which must also be traversed. Native allocations
    hidden inside third-party extensions are covered only by the RSS metric.
    """
    seen=set() if seen is None else seen
    if id(obj) in seen:return 0
    seen.add(id(obj));size=sys.getsizeof(obj)
    if isinstance(obj,dict):size+=sum(deep_size(k,seen)+deep_size(v,seen) for k,v in obj.items())
    elif isinstance(obj,(tuple,list,set,frozenset)):size+=sum(deep_size(v,seen) for v in obj)
    elif isinstance(obj,np.ndarray):
        if obj.base is not None:size+=deep_size(obj.base,seen)
        if obj.dtype.hasobject:size+=sum(deep_size(v,seen) for v in obj.flat)
    elif hasattr(obj,'__dict__'):size+=deep_size(vars(obj),seen)
    return size

def main():
    rng=np.random.default_rng(20260908);h=rng.normal(size=4096).astype(np.float32)
    state=StreamingFeatureState(h);state.update_and_get(0.)
    initial_bytes=state.state_array_bytes;initial_deep=deep_size(state)
    points=rng.normal(size=10000).astype(np.float32)
    process=psutil.Process();rss_start=process.memory_info().rss;rss_peak=rss_start
    start=time.perf_counter();count=0
    for _ in range(1000):
        for point in points:state.update_and_get(point)
        count+=len(points)
        if not np.isfinite(state.output).all():raise AssertionError('Nonfinite long-stream feature')
        rss_peak=max(rss_peak,process.memory_info().rss)
    elapsed=time.perf_counter()-start
    reference_errors=[];prefix_errors=[];worst_case={}
    for case in range(128):
        historical=rng.normal(size=int(rng.integers(1,500))).astype(np.float32)
        online=rng.normal(size=int(rng.integers(1,200))).astype(np.float32)
        if case%8==0:historical[:]=3
        if case%8==1:online[::7]=np.nan
        a,b=StreamingFeatureState(historical),StreamingFeatureState(historical)
        for point in online:
            actual=a.update_and_get(point).copy()
            _update.py_func(float(point),b.reference,b.baseline,b.thresholds,b.ring,b.ew,b.counters,b.output,b.parameters,b.alphas,b.lags)
            difference=float(np.max(abs(actual-b.output)))
            if difference>max(reference_errors,default=0):
                index=int(np.argmax(abs(actual-b.output)))
                worst_case={'case':case,'feature_index':index,'compiled':float(actual[index]),'python':float(b.output[index]),'step':int(a.counters[0])}
            reference_errors.append(difference)
        prefix=int(rng.integers(0,len(online)+1))
        first=np.array(list(replay(historical,online)))
        other=np.r_[online[:prefix],rng.normal(size=int(rng.integers(1,200))).astype(np.float32)*100]
        second=np.array(list(replay(historical,other)))
        prefix_errors.append(float(np.max(abs(first[:prefix]-second[:prefix]))) if prefix else 0.)
    report={'status':'PASS','scope':'SYNTHETIC_ENGINEERING_ONLY','points':count,'elapsed_seconds':elapsed,'feature_update_us_per_point':elapsed/count*1e6,
        'state_array_bytes_before':initial_bytes,'state_array_bytes_after':state.state_array_bytes,'deep_state_bytes_before':initial_deep,'deep_state_bytes_after':deep_size(state),
        'rss_start_bytes':rss_start,'rss_peak_bytes':rss_peak,'rss_end_bytes':process.memory_info().rss,
        'random_cases':128,'python_reference_max_abs_diff':max(reference_errors),'prefix_max_abs_diff':max(prefix_errors),
        'reference_tolerance':1e-5,'online_history_retained':8,'worst_case':worst_case}
    report['status']='PASS' if report['state_array_bytes_before']==report['state_array_bytes_after'] and report['python_reference_max_abs_diff']<1e-5 and report['prefix_max_abs_diff']==0 else 'FAIL'
    write_json(ROOT/'artifacts/validation/STREAMING_STRESS.json',report);print(report,flush=True)
    assert report['state_array_bytes_before']==report['state_array_bytes_after']
    assert report['python_reference_max_abs_diff']<1e-5 and report['prefix_max_abs_diff']==0

if __name__=='__main__':main()
