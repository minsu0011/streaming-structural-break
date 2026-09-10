"""Fixed-policy scalar-call, historical-init and 10M-point CPU measurements."""
from pathlib import Path
import gc
import hashlib
import json
import os
import time
import numpy as np
import pandas as pd
import psutil
from src.next.data import prepare_dev
from src.next.candidate_io import load_full_oof,model_class
from src.next.references import REFERENCES,load_reference_fold,verify_reference_sources
from src.utils.artifacts import write_json,sha256,utc_now


def assert_isolated(root):
    root = Path(root).resolve()
    excluded = {os.getpid(),*(process.pid for process in psutil.Process().parents())}
    active = []
    for process in psutil.process_iter(['pid','name']):
        if process.pid in excluded or not (process.info['name'] or '').lower().startswith('python'):
            continue
        try:
            if Path(process.cwd()).resolve()==root:
                active.append(process.pid)
        except (psutil.AccessDenied,psutil.NoSuchProcess):
            continue
    if active:
        raise RuntimeError(f'Isolated timing refused while other workspace Python workers are active: {active}')


def array_bytes_in_state(state):
    seen = set()
    def visit(value):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        if isinstance(value,np.ndarray):
            return value.nbytes
        if isinstance(value,dict):
            return sum(visit(v) for v in value.values())
        if isinstance(value,(tuple,list)):
            return sum(visit(v) for v in value)
        if hasattr(value,'__dict__'):
            return visit(vars(value))
        return 0
    return visit(state)


class NativeNextPrepared:
    def __init__(self,model):
        self.model = model
    def new_state(self,historical):
        return NativeNextDetector(self.model,historical)


class NativeNextDetector:
    def __init__(self,model,historical):
        self.model = model
        self.state = model.make_state(historical)
    def predict_one(self,point):
        return float(np.float32(self.model.predict_one(self.state.update(point))))
    @property
    def state_array_bytes(self):
        return array_bytes_in_state(self.state)


def load_benchmark_model(root,data,eid):
    root = Path(root)
    if eid in REFERENCES:
        frame,prediction,model = load_reference_fold(root,data.guard,eid,0)
        model_path = root/'artifacts/models/real'/REFERENCES[eid]/'fold_0.joblib'
        ids = frame.dataset_id.to_numpy()
    elif eid=='FAST_REFERENCE':
        from src.models.ar_calibrated import ARCalibratedBundle
        name = 'M4_lightgbm_ABCD_S3_122668fc55e8'
        model_path = root/'artifacts/models/real'/name/'fold_0.joblib'
        model = ARCalibratedBundle.load(model_path)
        oof = root/'data/processed/oof'/name
        manifest = json.loads((oof/'MANIFEST.json').read_text(encoding='utf-8'))
        cache = root/manifest['cache_directory']
        cache_manifest = json.loads((cache/'MANIFEST.json').read_text(encoding='utf-8'))
        if cache_manifest['split_sha256']!=data.guard.split.digest or sha256(cache/'fold_0.parquet')!=cache_manifest['fold_sha256']['0']:
            raise RuntimeError('Fast reference cache changed')
        frame = pd.read_parquet(cache/'fold_0.parquet',columns=['dataset_id','time_online','target'])
        prediction = np.load(oof/'fold_0.npy',allow_pickle=False)
        ids = frame.dataset_id.to_numpy()
        data.guard.record_rows(ids,purpose='fast reference benchmark coordinates')
    else:
        candidate,report,metadata,full_prediction,sources = load_full_oof(data,eid)
        rows = np.flatnonzero(metadata['fold']==0)
        prediction,ids = full_prediction[rows],metadata['dataset_id'][rows]
        model_path = root/'artifacts/next/experiments'/eid/'full/fold_0.joblib'
        model = model_class(candidate).load(model_path)
    cuts = np.r_[0,np.flatnonzero(np.diff(ids))+1,len(ids)]
    expected = {int(ids[a]):prediction[a:b] for a,b in zip(cuts[:-1],cuts[1:])}
    return model,model_path,expected


def benchmark_runtime(root,eid,variant,tag,prepare,runtime_receipt,driver_path,parity_report=None):
    root = Path(root)
    if not tag.replace('_','').isalnum() or not variant.replace('_','').isalnum():
        raise ValueError('Invalid benchmark tag or variant')
    assert_isolated(root)
    verify_reference_sources(root)
    data = prepare_dev(root)
    policy_path = root/'configs/next_latency_policy.json'
    policy = json.loads(policy_path.read_text(encoding='utf-8'))
    selected = data.guard.admit(policy['series_ids'],purpose='fixed held-out latency sample')
    if any(data.guard.split.assignment(sid)[1]!=0 for sid in selected):
        raise RuntimeError('Latency sample moved outside original held-out fold zero')
    # Preserve the preregistered SHA order, not the guard's sorted return order.
    selected = policy['series_ids']
    tick = time.perf_counter()
    model,model_path,expected = load_benchmark_model(root,data,eid)
    model_load_seconds = time.perf_counter()-tick
    receipt = runtime_receipt(model)
    if parity_report is not None:
        parity_path = root/parity_report
        parity = json.loads(parity_path.read_text(encoding='utf-8'))
        if parity['status']!='PASS' or not parity['all_float32_predictions_bitwise_equal'] or parity['points']!=4014405:
            raise RuntimeError('A completed full-DEV runtime parity report is required')
        actual_hash = parity['identity'].get('runtime_sha256',parity['identity'].get('runtime_source_sha256'))
        if actual_hash!=receipt['runtime_source_sha256']:
            raise RuntimeError('Optimized runtime changed after full-DEV parity')
    else:
        parity_path = None
    identity = {'candidate':eid,'variant':variant,'tag':tag,'model_kind':'CV_FOLD_0_MODEL',
        'model_sha256':sha256(model_path),'policy_sha256':sha256(policy_path),'runtime_receipt':receipt,
        'benchmark_source_sha256':sha256(Path(__file__)),'driver_sha256':sha256(driver_path),
        'parity_report_sha256':sha256(parity_path) if parity_path else None}
    out = root/'artifacts/next/latency'/tag
    out.mkdir(parents=True,exist_ok=True)
    report_path = out/(eid+'_'+variant+'.json')
    if report_path.exists():
        old = json.loads(report_path.read_text(encoding='utf-8'))
        if old['status']!='PASS' or old['identity']!=identity:
            raise RuntimeError('Completed benchmark identity changed; do not select timings by silent reruns')
        print('LATENCY retained PASS',eid,variant,flush=True)
        return old
    examples = [(sid,*data.get(sid)[:2]) for sid in selected]
    prepared = prepare(model)
    _,first_h,first_o = examples[0]
    tick = time.perf_counter()
    warm = prepared.new_state(first_h)
    warm.predict_one(first_o[0])
    cold_seconds = time.perf_counter()-tick
    del warm
    assert_isolated(root)
    point_ns,initialization,series_loop,series_lengths = [],[],[],[]
    sizes = []
    for sid,historical,points in examples:
        start = time.perf_counter_ns()
        detector = prepared.new_state(historical)
        initialized = time.perf_counter_ns()
        prediction = np.empty(len(points),dtype=np.float32)
        durations = np.empty(len(points),dtype=np.int64)
        loop_start = time.perf_counter_ns()
        for j,point in enumerate(points):
            before = time.perf_counter_ns()
            prediction[j] = detector.predict_one(point)
            durations[j] = time.perf_counter_ns()-before
        loop_end = time.perf_counter_ns()
        np.testing.assert_array_equal(prediction,expected[sid])
        initialization.append((initialized-start)/1e9)
        series_loop.append((loop_end-loop_start)/1e9)
        series_lengths.append(len(points))
        point_ns.append(durations)
        sizes.append(detector.state_array_bytes)
    durations = np.concatenate(point_ns)/1000.
    total_real_points = sum(series_lengths)
    amortized = (sum(initialization)+sum(series_loop))/total_real_points*1e6
    buffer = np.concatenate([points for _,_,points in examples])[:policy['block_points']]
    if len(buffer)!=policy['block_points']:
        raise RuntimeError('Insufficient real values for fixed continuous-stream buffer')
    detector = prepared.new_state(first_h)
    detector.predict_one(buffer[0])
    gc.collect()
    before_size = detector.state_array_bytes
    process = psutil.Process()
    rss_before = process.memory_info().rss
    peak_rss = rss_before
    processed,elapsed,million = 0,0.,None
    digest = hashlib.sha256()
    wall_start = time.perf_counter()
    while processed<policy['continuous_points']:
        block = buffer[:min(len(buffer),policy['continuous_points']-processed)]
        tick = time.perf_counter_ns()
        prediction = np.asarray([detector.predict_one(point) for point in block],dtype=np.float32)
        elapsed += (time.perf_counter_ns()-tick)/1e9
        if not np.isfinite(prediction).all() or np.any((prediction<0)|(prediction>1)):
            raise RuntimeError('Long-stream output range failure')
        digest.update(prediction.tobytes())
        processed += len(block)
        if processed==policy['measured_checkpoint_points']:
            million = elapsed
        if processed%1000000==0:
            assert_isolated(root)
            peak_rss = max(peak_rss,process.memory_info().rss)
            print('NEXT LATENCY',eid,variant,processed,round(elapsed,3),'seconds',flush=True)
    if detector.state_array_bytes!=before_size:
        raise RuntimeError('Long-stream online state grew')
    report = {'status':'PASS','identity':identity,'created_utc':utc_now(),'model_load_seconds':model_load_seconds,
        'first_history_and_jit_seconds':cold_seconds,'isolated_from_workspace_research_workers':True,
        'point_distribution_us':{'samples':len(durations),'median':float(np.median(durations)),
            'p95':float(np.quantile(durations,.95)),'p99':float(np.quantile(durations,.99)),
            'mean':float(np.mean(durations)),'includes_per_point_timer_and_float32_store':True},
        'real_history_initialization':{'series':len(examples),'mean_seconds':float(np.mean(initialization)),
            'median_seconds':float(np.median(initialization)),'p95_seconds':float(np.quantile(initialization,.95)),
            'p99_seconds':float(np.quantile(initialization,.99)),'historical_length_min':min(len(h) for _,h,_ in examples),
            'historical_length_max':max(len(h) for _,h,_ in examples),'actual_online_points':total_real_points,
            'aggregate_h_and_online_seconds':sum(initialization)+sum(series_loop),'amortized_us_per_point':amortized,
            'one_million_amortized_seconds':amortized,'ten_million_amortized_seconds':10*amortized},
        'continuous_stream':{'points':processed,'measured_prediction_seconds':elapsed,
            'measured_first_million_seconds':million,'microseconds_per_point':elapsed/processed*1e6,
            'wall_seconds_including_checks':time.perf_counter()-wall_start,'output_sha256':digest.hexdigest(),
            'state_array_bytes_before':before_size,'state_array_bytes_after':detector.state_array_bytes,
            'rss_before_bytes':rss_before,'rss_peak_bytes':peak_rss,'rss_after_bytes':process.memory_info().rss},
        'real_prefix_state_array_bytes_maximum':max(sizes),'all_sampled_heldout_predictions_bitwise_equal':True,
        'seal_rows':0,'reduced_new_usage':0,'quality_score_computed_from_continuous_stream':False,
        'limitations':'CPU timings omit file/socket/cloud overhead. Continuous input is a bounded real-value buffer repeated into one synthetic-duration stream; quality is not scored there. Amortized 1M/10M estimates assume the fixed sample H/online-length mixture.'}
    write_json(report_path,report)
    print('NEXT LATENCY PASS',eid,variant,'median_us',report['point_distribution_us']['median'],'amortized_us',amortized,flush=True)
    return report
