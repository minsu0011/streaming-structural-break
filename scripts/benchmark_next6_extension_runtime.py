"""Final scalar benchmark after heavy owned research jobs finish."""
from pathlib import Path
import sys,os,argparse,time,json,subprocess,platform
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next6_extension.common import BASE,save,read_json,digest,utc_now,activity,csv_write
METHODS=('M0','M3','FRESH_CORE20','OBSERVABLE_READABLE27','OBSERVABLE_FUSED27')

def workload():
    from src.next2.counterexamples import generate
    return generate('N2',63,partition=37,horizon=4096,control=True,lock=read_json(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json'))[:2]

def prepared(method):
    import joblib
    if method=='M0':
        from src.next5.runtime import SharedResearchBaseline
        from src.next5.model import FeatureModel
        models=[FeatureModel.load(ROOT/f'artifacts/next5/references/fits/PRIMARY/full/{kind}/fold_0/model.joblib') for kind in ('J102','C205')]
        return SharedResearchBaseline(*models)
    if method=='M3':
        from src.next6.runtime import Prepared
        return Prepared(joblib.load(ROOT/'artifacts/next6/fold_0/outer/M3/model.joblib'))
    from src.next6_extension.observer import load
    model=load(BASE/'observable_models/fold_0/model.joblib')
    if method=='OBSERVABLE_READABLE27':return model
    if method=='OBSERVABLE_FUSED27':
        from src.next6_extension.validated_runtime import Prepared
        return Prepared(model)
    from src.next6_extension.fresh_runtime import SafeState
    class Core:
        def new_state(self,h):return SafeState(h,threshold=model.threshold,graph=model.graph)
    return Core()

def memory(method,state):
    from src.next.fused_runtime import unique_array_bytes
    if method=='M0':return state.state_array_bytes,None
    if method=='M3':return unique_array_bytes(state.args[:12]+(state.args[16],state.args[17],state.args[18],state.args[20])),None
    feature=state if method=='FRESH_CORE20' else state.features;arrays=feature.args[:11]+(feature.args[14],feature.H_block_z,feature.H_previous_state)
    if method=='OBSERVABLE_FUSED27':arrays=arrays+(state.args[3],state.args[4])
    return unique_array_bytes(arrays),state.bounded_mutable_state_bytes

def cold(method):
    import numpy as np
    h,o=workload();start=time.perf_counter();p=prepared(method);prep=time.perf_counter()-start;start=time.perf_counter();state=p.new_state(h);initial=time.perf_counter()-start;start=time.perf_counter();out=state.predict_one(float(o[0]));first=time.perf_counter()-start;assert np.isfinite(out).all()
    print(json.dumps({'method':method,'model_load_and_prepare_seconds':prep,'cold_H_state_seconds':initial,'first_scalar_seconds':first,'fresh_process_and_empty_numba_disk_cache':True,'synthetic_generation_before_timer':True,'shared_cross_method_JIT_cache':False}))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cold-method',choices=METHODS);parser.add_argument('--lock-only',action='store_true');a=parser.parse_args()
    if a.cold_method:cold(a.cold_method);return
    lock=BASE/'LATENCY_WORKLOAD_LOCK.json'
    if not lock.exists():save(lock.name,{'created_utc':utc_now(),'source_sha256':digest(Path(__file__)),'methods':METHODS,'case':'N2','partition':37,'seed':63,'H_length':2048,'online_length':4096,'warm_repetitions':7,'rotation':'cyclic method order, new H state per repetition','cold':'separate process and empty NumPy/Numba cache directory per method; report process costs separately','timing':'actual scalar API calls using perf_counter_ns, no output validation inside timed calls','no_change_to_models':True})
    assert read_json(lock)['source_sha256']==digest(Path(__file__))
    if a.lock_only:return
    assert all((BASE/f'graph_bootstrap/fold_{f}/DONE.json').exists() and (BASE/f'extended_synthetic/fold_{f}/DONE.json').exists() for f in range(5)),'Heavy owned research jobs must finish first'
    assert not (BASE/'LATENCY_NEXT6_EXTENSION.json').exists(),'Preserve completed benchmark'
    os.environ['NUMBA_CACHE_DIR']=str(BASE/'numba_cache');import numpy as np,psutil
    with activity('final_cold_and_hot_scalar_latency','FINAL_AUDIT'):
        coldrows=[]
        for method in METHODS:
            cache=BASE/'runtime'/('cold_latency_'+method);assert not cache.exists();cache.mkdir(parents=True);env=os.environ.copy();env['NUMBA_CACHE_DIR']=str(cache);start=time.perf_counter();r=subprocess.run([sys.executable,'-B',str(Path(__file__)),'--cold-method',method],cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8');(BASE/f'audit/COLD_LATENCY_{method}.log').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8');assert r.returncode==0,r.stderr;row=json.loads(r.stdout);row['subprocess_wall_seconds']=time.perf_counter()-start;coldrows.append(row)
        h,o=workload();models={m:prepared(m) for m in METHODS};measure={m:[] for m in METHODS};block={m:[] for m in METHODS};initial={m:[] for m in METHODS};sizes={}
        for m,p in models.items():p.new_state(h).predict_one(float(o[0]))
        cpu=psutil.cpu_percent(interval=.2)
        for repeat in range(7):
            order=METHODS[repeat%len(METHODS):]+METHODS[:repeat%len(METHODS)]
            for method in order:
                start=time.perf_counter_ns();state=models[method].new_state(h);initial[method].append((time.perf_counter_ns()-start)/1e9);start=time.perf_counter_ns()
                for value in o:
                    tick=time.perf_counter_ns();result=state.predict_one(float(value));measure[method].append(time.perf_counter_ns()-tick)
                block[method].append((time.perf_counter_ns()-start)/len(o));assert np.isfinite(result).all();sizes[method]=memory(method,state)
        out=[];reference=np.median(measure['M0'])
        for method in METHODS:
            q=np.quantile(measure[method],[.5,.95,.99]);out.append({'method':method,'points':len(measure[method]),'scalar_median_us':float(q[0]/1000),'scalar_p95_us':float(q[1]/1000),'scalar_p99_us':float(q[2]/1000),'median_ratio_vs_M0':float(q[0]/reference),'median_block_average_us':float(np.median(block[method])/1000),'warm_H_initialization_seconds':float(np.median(initial[method])),'instance_array_bytes_excluding_shared_model_including_H':int(sizes[method][0]),'mutable_stream_numeric_bytes_when_available':sizes[method][1],'python_object_overhead_excluded':True})
        csv_write('LATENCY_NEXT6_EXTENSION.csv',out);csv_write('COLD_START_NEXT6_EXTENSION.csv',coldrows);save('LATENCY_NEXT6_EXTENSION.json',{'status':'COMPLETE','created_utc':utc_now(),'rows':out,'cold':coldrows,'same_H_and_stream':True,'heavy_owned_jobs_finished':True,'system_cpu_percent_before_hot':cpu,'platform':platform.platform(),'logical_cpus':psutil.cpu_count(),'physical_memory_bytes':psutil.virtual_memory().total,'no_hardware_independent_guarantee':True,'fresh_vs_readable_same_output_math':True,'M0_M3_vs_fresh_are_different_algorithms':True,'workload_lock_sha256':digest(lock)})
if __name__=='__main__':main()
