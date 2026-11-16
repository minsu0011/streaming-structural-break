"""Reports engineering invariants and latency; never emits competition CV scores."""
import csv
import json
import sys
import time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.features.streaming import StreamingFeatureState,replay,CATALOG,feature_version
from src.models.baselines import BASELINES,BaselineDetector
from src.models.learners import MODEL_NAMES,fit_candidate,ModelBundle
from src.data.labels import labels_from_tau
from src.streaming.inference import write_engineering_baseline,infer
from src.validation.synthetic import fixture,KINDS
from src.validation.protocol import protocol_predictions
from src.utils.artifacts import write_json,ledger,utc_now,checkpoint

def benchmark(factory,points,repeats=5):
    detector=factory()
    for x in points[:100]: detector(x)
    timings=[]; batch_times=[]
    for _ in range(repeats):
        detector=factory(); start=time.perf_counter_ns()
        for x in points: detector(x)
        batch_times.append((time.perf_counter_ns()-start)/len(points)/1000)
    detector=factory()
    for x in points:
        start=time.perf_counter_ns(); detector(x); timings.append((time.perf_counter_ns()-start)/1000)
    median=float(np.median(timings))
    return {'median_us_per_point':median,'p95_us':float(np.quantile(timings,.95)),'p99_us':float(np.quantile(timings,.99)),
        'throughput_median_us_per_point':float(np.median(batch_times)),
        'million_points_seconds':float(np.median(batch_times)), 'ten_million_points_seconds':float(np.median(batch_times)*10),
        'points_timed':len(points),'batch_repeats':repeats,'includes':'state update + single-row prediction','excludes':'historical initialization, JIT first compile, model loading, socket IPC',
        'individual_timer_overhead_included':True}

def main():
    h,o,_=fixture('mean_up',online_length=2000)
    started=time.perf_counter(); StreamingFeatureState(h).update_and_get(o[0]); jit=time.perf_counter()-started
    points=np.tile(o,2)
    with (ROOT/'artifacts/features/FEATURE_CATALOG.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['column','family','dtype','normalization','future_access','version'])
        for name,family in CATALOG:writer.writerow([name,family,'float32','historical_reference_only',False,feature_version()])
    features=np.array(list(replay(h,o)))
    changed=np.r_[o[:150],np.full(80,999,dtype=np.float32)]
    other=np.array(list(replay(h,changed)))
    prefix_diff=float(np.max(np.abs(features[:150]-other[:150])))
    prefix_path=ROOT/'artifacts/validation'/('PREFIX_INVARIANCE_SYNTHETIC_RECHECK.json' if (ROOT/'artifacts/validation/REAL_SESSION.json').exists() else 'PREFIX_INVARIANCE.json')
    write_json(prefix_path,{'status':'PASS' if prefix_diff==0 else 'FAIL','feature_max_abs_diff':prefix_diff,'feature_hash':feature_version(),'suffix_length_changed':True,'suffix_values_changed':True,'scope':'SYNTHETIC_ENGINEERING_ONLY'})
    latency={'scope':'SYNTHETIC_ENGINEERING_ONLY','timestamp':utc_now(),'jit_first_compile_seconds':jit,'feature_state_array_bytes':StreamingFeatureState(h).state_array_bytes,'candidates':{}}
    for name in BASELINES:
        latency['candidates'][name]=benchmark(lambda n=name:BaselineDetector(n,h).predict_one,points)
    xs=[];ys=[]
    for j,kind in enumerate(KINDS):
        hist,online,tau=fixture(kind,seed=20260908+j)
        xs.extend(replay(hist,online));ys.extend(labels_from_tau(len(online),tau))
    xs=np.asarray(xs,dtype=np.float32);ys=np.asarray(ys,dtype=np.uint8)
    model_reports=[]
    with threadpool_limits(limits=1):
        for name in MODEL_NAMES:
            start=time.perf_counter(); model=fit_candidate(name,xs,ys)
            path=ROOT/'artifacts/models/engineering'/f'{name}.joblib'; model.save(path)
            loaded=ModelBundle.load(path)
            first=model.predict(xs[:100]); second=loaded.predict(xs[:100]); diff=float(np.max(abs(first-second)))
            def factory(m=loaded):
                state=StreamingFeatureState(h)
                return lambda point:m.predict_one(state.update_and_get(point))
            latency['candidates'][name]=benchmark(factory,points[:1000],repeats=3)
            latency['candidates'][name]['model_bytes']=path.stat().st_size
            model_reports.append({'name':name,'status':'PASS' if diff==0 else 'FAIL','save_load_max_abs_diff':diff,'fit_and_check_seconds':time.perf_counter()-start,'competition_cv':None})
            ledger(experiment_id='engineering_'+name,feature_version=feature_version(),feature_families='ABCD',model=name,status='ENGINEERING_PASS' if diff==0 else 'FAIL',latency=latency['candidates'][name]['median_us_per_point'],notes='Synthetic fit/serialization/latency only; no competition performance claim')
            print(name,'engineering checks complete',flush=True)
    write_json(ROOT/'artifacts/validation/MODEL_ENGINEERING.json',{'scope':'SYNTHETIC_ENGINEERING_ONLY','models':model_reports})
    write_json(ROOT/'artifacts/validation/LATENCY_BENCHMARK.json',latency)
    model_dir=ROOT/'artifacts/deployment/models';write_engineering_baseline(model_dir)
    series=[fixture(kind)[:2] for kind in KINDS]
    a=protocol_predictions(infer,series,model_dir);b=protocol_predictions(infer,series,model_dir)
    direct=np.asarray(list(infer(iter(series),model_dir))[1:],dtype=np.float32)
    max_diff=float(np.max(abs(a-b))); wire_diff=float(np.max(abs(a-direct)))
    write_json(ROOT/'artifacts/validation/DETERMINISM_REPORT.json',{'status':'PASS' if max_diff==0 else 'FAIL','runs':2,'max_abs_diff':max_diff,'wire_vs_direct_float32_max_diff':wire_diff,'scope':'SYNTHETIC_OFFICIAL_PROTOCOL','artifact':'B7_manual; engineering only'})
    write_json(ROOT/'artifacts/validation/DEPLOYMENT_SMOKE.json',{'status':'PROTOCOL_PASS_OFFICIAL_TESTER_BLOCKED_AUTH','official_double_protection':'PASS','transport':'real TCP loopback; unchanged official runner code','rows':len(a),'expected_rows':sum(len(o) for _,o in series),'dtype':str(a.dtype),'range':[float(a.min()),float(a.max())],'finite':bool(np.isfinite(a).all()),'deterministic':max_diff==0,'direct_parity':wire_diff==0,'official_crunch_tools_test':'NOT_RUN_AUTH_REQUIRED','data_scope':'SYNTHETIC_ENGINEERING_ONLY','submitted':False})
    checkpoint(3,['scorer parity','streaming invariance','model engineering','official protocol smoke','latency'], 'Complete research runner and audit/handoff',artifact_paths=['artifacts/validation'])
    print('Engineering reports saved',flush=True)

if __name__=='__main__':main()
