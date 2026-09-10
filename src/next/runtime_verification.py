"""Full-DEV output verification shared by later runtime specializations."""
from pathlib import Path
import gc
import json
import time
import numpy as np
from src.next.data import prepare_dev
from src.next.candidate_io import load_full_oof,model_class
from src.next.progress import checkpoint
from src.utils.artifacts import write_json,sha256,utc_now


def verify_runtime(root,eid,variant,prepare,runtime_hash,driver_path):
    root = Path(root)
    if not variant.replace('_','').isalnum():
        raise ValueError('Invalid runtime variant identifier')
    data = prepare_dev(root)
    candidate,original,metadata,prediction,sources = load_full_oof(data,eid)
    out = root/'artifacts/next/runtime_parity'/eid/variant
    out.mkdir(parents=True,exist_ok=True)
    identity = {'candidate':candidate,'variant':variant,'oof_sources':sources,
        'runtime_sha256':runtime_hash(),'driver_sha256':sha256(driver_path),
        'verification_source_sha256':sha256(Path(__file__)),'candidate_io_sha256':sha256(root/'src/next/candidate_io.py')}
    path = out/'POLICY.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=identity:
        raise RuntimeError('Runtime verification identity changed')
    if not path.exists():
        write_json(path,identity)
    rows = []
    for fold in range(5):
        result_path = out/f'fold_{fold}.json'
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding='utf-8'))
            if result['status']!='PASS':
                raise RuntimeError('A previous runtime fold failed')
            rows.append(result)
            continue
        model = model_class(candidate).load(root/'artifacts/next/experiments'/eid/'full'/f'fold_{fold}.joblib')
        prepared = prepare(model)
        tick = time.perf_counter()
        series,points_seen,max_bytes = 0,0,0
        for j,sid in enumerate(data.ids):
            if data.series['fold'][j]!=fold:
                continue
            historical,points,_ = data.get(sid)
            detector = prepared.new_state(historical)
            size = detector.state_array_bytes
            actual = np.asarray([detector.predict_one(point) for point in points],dtype=np.float32)
            a,b = data.series['online_offsets'][j:j+2]
            np.testing.assert_array_equal(actual,prediction[a:b])
            if detector.state_array_bytes!=size:
                raise RuntimeError('Runtime retained growing online state')
            series += 1
            points_seen += len(points)
            max_bytes = max(max_bytes,size)
            if series%500==0:
                print('RUNTIME PARITY',eid,variant,fold,series,round(time.perf_counter()-tick,1),flush=True)
        result = {'status':'PASS','fold':fold,'series':series,'points':points_seen,'maximum_absolute_difference':0.,
            'all_float32_predictions_bitwise_equal':True,'fixed_memory':True,'state_array_bytes_maximum':max_bytes,
            'elapsed_seconds':time.perf_counter()-tick,'created_utc':utc_now(),'seal_rows':0}
        write_json(result_path,result)
        rows.append(result)
        del prepared,model,detector
        gc.collect()
    if sum(row['series'] for row in rows)!=8000 or sum(row['points'] for row in rows)!=len(prediction):
        raise RuntimeError('Incomplete runtime verification coverage')
    write_json(out/'RESULTS.json',{'status':'PASS','candidate':eid,'variant':variant,'identity':identity,'folds':rows,
        'series':8000,'points':len(prediction),'all_float32_predictions_bitwise_equal':True,
        'maximum_absolute_difference':0.,'seal_rows':0,'reduced_new_usage':0,'created_utc':utc_now(),
        'timing_interpretation':'Engineering verification, potentially concurrent with research; not an isolated latency measurement.'})
    checkpoint(root,eid+'/'+variant,[eid+' '+variant+' full DEV prediction parity'],
        ['isolated_runtime_benchmark','remaining_research_backlog'],artifacts=[str((out/'RESULTS.json').relative_to(root))],
        best=max(.611292686371357,original['full']['mean']))
    print('RUNTIME PARITY COMPLETE',eid,variant,len(prediction),flush=True)

