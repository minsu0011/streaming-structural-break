"""Isolated component timing on the same fixed 128 held-out real sequences."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.data import prepare_dev
from src.next.runtime_benchmark import assert_isolated,load_benchmark_model
from src.next.runtime_components import PreparedComponents,source_hash
from src.next.references import verify_reference_sources
from src.utils.artifacts import write_json,sha256,utc_now


def distribution(values):
    values = np.asarray(values,dtype=float)/1000.
    return {'median_us':float(np.median(values)),'p95_us':float(np.quantile(values,.95)),
        'p99_us':float(np.quantile(values,.99)),'mean_us':float(np.mean(values)),
        'estimated_one_million_seconds':float(np.mean(values)),'estimated_ten_million_seconds':10.*float(np.mean(values))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate',required=True)
    parser.add_argument('--combined-runtime',required=True)
    args = parser.parse_args()
    assert_isolated(ROOT)
    verify_reference_sources(ROOT)
    policy_path = ROOT/'configs/next_latency_policy.json'
    policy = json.loads(policy_path.read_text(encoding='utf-8'))
    data = prepare_dev(ROOT)
    data.guard.admit(policy['series_ids'],purpose='fixed component-timing IDs')
    model,path,expected = load_benchmark_model(ROOT,data,args.candidate)
    prepared = PreparedComponents(model)
    combined_path = ROOT/'artifacts/next/latency/research_v1'/f'{args.candidate}_{args.combined_runtime}.json'
    combined = json.loads(combined_path.read_text(encoding='utf-8'))
    if combined['status']!='PASS' or combined['identity']['model_sha256']!=sha256(path) or combined['identity']['policy_sha256']!=sha256(policy_path):
        raise RuntimeError('Matching completed combined benchmark required')
    parity_path = None
    if prepared.is_next:
        parity_path = ROOT/'artifacts/next/runtime_parity'/args.candidate/'separate_feature_tree_calls/RESULTS.json'
        parity = json.loads(parity_path.read_text(encoding='utf-8'))
        if parity['status']!='PASS' or parity['points']!=4014405 or parity['identity']['runtime_sha256']!=source_hash():
            raise RuntimeError('Full-DEV component-runtime parity must precede timing')
    out = ROOT/'artifacts/next/latency/components_v1'
    out.mkdir(exist_ok=True)
    identity = {'candidate':args.candidate,'model_sha256':sha256(path),'policy_sha256':sha256(policy_path),
        'runtime_source_sha256':source_hash(),'driver_sha256':sha256(Path(__file__)),
        'combined_report_sha256':sha256(combined_path),'parity_sha256':sha256(parity_path) if parity_path else None}
    report_path = out/(args.candidate+'.json')
    if report_path.exists():
        old = json.loads(report_path.read_text(encoding='utf-8'))
        if old['status']!='PASS' or old['identity']!=identity:
            raise RuntimeError('Completed component benchmark changed; no silent timing selection')
        print('COMPONENT TIMING retained PASS',args.candidate,flush=True)
        return
    h,o,_ = data.get(policy['series_ids'][0])
    warm = prepared.new_state(h)
    warm.predict_one(o[0])
    feature_ns,model_ns = [],[]
    for sid in policy['series_ids']:
        assert_isolated(ROOT)
        h,o,_ = data.get(sid)
        state = prepared.new_state(h)
        prediction = np.empty(len(o),dtype=np.float32)
        for j,point in enumerate(o):
            start = time.perf_counter_ns()
            state.feature_update(point)
            middle = time.perf_counter_ns()
            prediction[j] = state.model_predict()
            end = time.perf_counter_ns()
            feature_ns.append(middle-start)
            model_ns.append(end-middle)
        np.testing.assert_array_equal(prediction,expected[sid])
        # The tree-only call must not advance causal state.
        assert np.float32(state.model_predict())==prediction[-1]
        assert np.float32(state.model_predict())==prediction[-1]
    report = {'status':'PASS','identity':identity,'created_utc':utc_now(),'series':128,'points':len(feature_ns),
        'feature_state_update':distribution(feature_ns),'model_prediction':distribution(model_ns),
        'combined_production_point_distribution_us':combined['point_distribution_us'],
        'historical_initialization':combined['real_history_initialization'],
        'all_sampled_heldout_predictions_bitwise_equal':True,'model_only_call_does_not_advance_state':True,
        'seal_rows':0,'reduced_new_usage':0,
        'interpretation':'Separate scalar feature and tree calls use the exact production calculations with an extra dispatch boundary. Their sum includes two timer/call costs and need not equal the separately measured fused combined call. Component 1M/10M figures are mean-call extrapolations; combined 10M was directly measured.'}
    write_json(report_path,report)
    print('COMPONENT TIMING PASS',args.candidate,report['feature_state_update']['median_us'],report['model_prediction']['median_us'],flush=True)


if __name__=='__main__':
    main()
