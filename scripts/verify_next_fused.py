"""Verify a prepared runtime against every saved primary-fold DEV prediction."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import gc
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.data import prepare_dev
from src.next.candidate_io import load_full_oof,model_class
from src.next.fused_runtime import PreparedNextPredictor,source_hash
from src.next.progress import checkpoint
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate',required=True)
    parser.add_argument('--mode',choices=['compiled','resident'],default='resident')
    args = parser.parse_args()
    started = time.perf_counter()
    data = prepare_dev(ROOT)
    candidate,original,metadata,prediction,sources = load_full_oof(data,args.candidate)
    out = ROOT/'artifacts/next/fused_parity'/(args.candidate+'_'+args.mode)
    out.mkdir(parents=True,exist_ok=True)
    identity = {'candidate':candidate,'mode':args.mode,'oof_sources':sources,'runtime_source_sha256':source_hash(),
        'script_sha256':sha256(Path(__file__)),'candidate_io_sha256':sha256(ROOT/'src/next/candidate_io.py'),
        'scope':'All 8000 DEV series, each with its original held-out fold model; exactly one chronological update per point.'}
    path = out/'POLICY.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=identity:
        raise RuntimeError('Prepared parity code or model identity changed')
    if not path.exists():
        write_json(path,identity)
    rows = []
    for fold in range(5):
        result_path = out/f'fold_{fold}.json'
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding='utf-8'))
            if result['status']!='PASS':
                raise RuntimeError('Previous parity fold did not pass')
            rows.append(result)
            print('FUSED PARITY RESUME',args.candidate,fold,flush=True)
            continue
        model_path = ROOT/'artifacts/next/experiments'/args.candidate/'full'/f'fold_{fold}.joblib'
        model = model_class(candidate).load(model_path)
        prepared = PreparedNextPredictor(model,resident=args.mode=='resident')
        tick = time.perf_counter()
        count,points_seen,largest_state = 0,0,0
        for j,sid in enumerate(data.ids):
            if data.series['fold'][j]!=fold:
                continue
            h,points,_ = data.get(sid)
            detector = prepared.new_state(h)
            before = detector.state_array_bytes
            values = np.asarray([detector.predict_one(point) for point in points],dtype=np.float32)
            start,end = data.series['online_offsets'][j:j+2]
            np.testing.assert_array_equal(values,prediction[start:end])
            if before!=detector.state_array_bytes:
                raise RuntimeError('Prepared online state grew')
            largest_state = max(largest_state,before)
            count += 1
            points_seen += len(points)
            if count%400==0:
                print('FUSED PARITY',args.candidate,fold,count,points_seen,round(time.perf_counter()-tick,1),flush=True)
        result = {'status':'PASS','candidate':args.candidate,'fold':fold,'series':count,'points':points_seen,
            'all_float32_predictions_bitwise_equal':True,'maximum_absolute_difference':0.,'state_array_bytes_maximum':largest_state,
            'fixed_memory_for_every_series':True,'elapsed_seconds':time.perf_counter()-tick,'timestamp':utc_now(),
            'model_sha256':sha256(model_path),'seal_rows':0,'reduced_new_usage':0}
        write_json(result_path,result)
        rows.append(result)
        del prepared,model,detector
        gc.collect()
    if sum(row['series'] for row in rows)!=8000 or sum(row['points'] for row in rows)!=len(prediction):
        raise RuntimeError('Incomplete full DEV parity coverage')
    write_json(out/'RESULTS.json',{'status':'PASS','candidate':args.candidate,'mode':args.mode,'identity':identity,
        'series':sum(row['series'] for row in rows),'points':sum(row['points'] for row in rows),
        'all_float32_predictions_bitwise_equal':True,'maximum_absolute_difference':0.,
        'elapsed_seconds':time.perf_counter()-started,'folds':rows,'seal_rows':0,'reduced_new_usage':0,
        'not_a_latency_benchmark':'Work may overlap other DEV fitting; isolated performance timing is a separate task.'})
    checkpoint(ROOT,args.candidate+'/prepared_runtime',[args.candidate+' all-DEV prepared prediction parity'],
        ['isolated_latency','historical_initialization_optimization','remaining_research_backlog'],
        artifacts=[str((out/'RESULTS.json').relative_to(ROOT))],best=max(.611292686371357,original['full']['mean']))
    print('FULL FUSED PARITY PASS',args.candidate,len(prediction),flush=True)


if __name__=='__main__':
    main()
