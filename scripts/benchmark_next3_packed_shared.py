"""Three prospective paired trials of the two fixed exact packed representations."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
import sys
import time
import hashlib
import gc
import numpy as np
import pandas as pd
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next3.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next3.packed_shared_runtime import PreparedPackedSharedPredictor,PackedSharedParts,source_hash,accounting,VARIANTS
from scripts.verify_next3_packed_shared import checked_implementation
from src.next2.shared_blend_runtime import PreparedSharedBlendPredictor,SharedParts
from src.next2.shared_inference import load_artifact
from src.next.data import prepare_dev
from scripts.benchmark_next2_runtime import assert_isolated,distribution


def trial(engine,variant,repeat,examples,buffer,expected,out):
    assert_isolated()
    directory=out/f'trial_{repeat}_{variant}';directory.mkdir(parents=True,exist_ok=False)
    atomic_json(directory/'START.json',{'created_utc':utc_now(),'variant':variant,'paired_trial':repeat},immutable=True)
    _,h0,o0=examples[0]
    tick=time.perf_counter();warm=engine.new_state(h0);warm.predict_one(o0[0]);warm_seconds=time.perf_counter()-tick
    points=[];histories=[];records=[];samples={}
    for inner in range(3):
        assert_isolated()
        for sid,h,o in examples:
            tick=time.perf_counter_ns();state=engine.new_state(h);initialization=time.perf_counter_ns()-tick
            values=np.empty(len(o),dtype=np.float32);duration=np.empty(len(o),dtype=np.int64)
            started=time.perf_counter_ns()
            for j,value in enumerate(o):
                tick=time.perf_counter_ns();values[j]=state.predict_one(value);duration[j]=time.perf_counter_ns()-tick
            elapsed=time.perf_counter_ns()-started
            if inner==0:samples[sid]=values
            else:np.testing.assert_array_equal(samples[sid],values)
            points.extend(duration);histories.append(initialization)
            records.append({'inner_repeat':inner,'dataset_id':sid,'historical_points':len(h),'online_points':len(o),
                'historical_init_ns':initialization,'online_loop_ns':elapsed,'point_median_ns':float(np.median(duration))})
        progress('Exact shared runtime timing',f'trial {repeat} {variant}: real-prefix repetition {inner+1}/3',experiment='E002_PACKED_SHARED')
    sample_hash=hashlib.sha256(b''.join(samples[sid].tobytes() for sid,_,_ in examples)).hexdigest()
    assert sample_hash==expected['sample_prediction_sha256']
    assert_isolated()
    parts_type=SharedParts if variant=='FROZEN_SHARED' else PackedSharedParts
    parts=parts_type(engine.new_state(h0));parts.feature_update(o0[0]);parts.model_predict()
    features=[];trees=[]
    for sid,h,o in examples:
        parts=parts_type(engine.new_state(h))
        for j,value in enumerate(o):
            tick=time.perf_counter_ns();parts.feature_update(value);middle=time.perf_counter_ns();pred=parts.model_predict();end=time.perf_counter_ns()
            features.append(middle-tick);trees.append(end-middle);assert pred==samples[sid][j]
    assert_isolated()
    state=engine.new_state(h0);memory,_,_=accounting(state);before=state.state_array_bytes
    gc.collect();process=psutil.Process();rss=process.memory_info().rss;peak=rss
    elapsed=0.;count=0;blocks=[];fingerprint=hashlib.sha256();wall=time.perf_counter();first_million=None
    while count<10_000_000:
        tick=time.perf_counter_ns();prediction=np.asarray([state.predict_one(v) for v in buffer],dtype=np.float32)
        duration=(time.perf_counter_ns()-tick)/1e9;elapsed+=duration;count+=len(buffer)
        assert np.isfinite(prediction).all();fingerprint.update(prediction.tobytes())
        blocks.append({'points':count,'seconds':duration,'cumulative_seconds':elapsed})
        if count==1_000_000:first_million=elapsed
        if count%1_000_000==0:
            assert_isolated();peak=max(peak,process.memory_info().rss)
            progress('Exact shared runtime direct 10M',f'trial {repeat} {variant}: {count} points, {elapsed:.3f} prediction seconds',experiment='E002_PACKED_SHARED')
            print('NEXT3 EXACT TIMING',repeat,variant,count,round(elapsed,3),flush=True)
    assert count==10_000_000 and state.state_array_bytes==before
    assert fingerprint.hexdigest()==expected['continuous_stream']['output_sha256']
    pd.DataFrame(records).to_csv(directory/'SERIES_TIMING.csv',index=False)
    pd.DataFrame(blocks).to_csv(directory/'CONTINUOUS_BLOCKS.csv',index=False)
    result={'status':'PASS','created_utc':utc_now(),'trial':repeat,'variant':variant,
        'point_distribution':distribution(points),'historical_initialization':distribution(histories),
        'feature_state_update':distribution(features),'model_prediction':distribution(trees),
        'warm_history_plus_possible_JIT_seconds':warm_seconds,'numeric_memory':memory,
        'continuous_stream':{'points':count,'measured_prediction_seconds':elapsed,'measured_first_million_seconds':first_million,
            'wall_seconds_including_checks':time.perf_counter()-wall,'output_sha256':fingerprint.hexdigest(),
            'state_array_bytes_before':before,'state_array_bytes_after':state.state_array_bytes,'rss_before_bytes':rss,
            'rss_peak_bytes':peak,'rss_after_bytes':process.memory_info().rss},
        'sample_prediction_sha256':sample_hash,'new_fits':0,'seal_rows':0,
        'files_sha256':{name:digest(directory/name) for name in ('SERIES_TIMING.csv','CONTINUOUS_BLOCKS.csv')}}
    atomic_json(directory/'RESULTS.json',result,immutable=True)
    return result


def main():
    assert_isolated(); implementation_sha = checked_implementation()
    hyp = BASE/'hypotheses/E002_PACKED_SHARED.json'; proofs = {}
    for stage in ('all_dev', 'oof', 'alternate'):
        path = BASE/'engineering/packed_shared'/stage/'RESULTS.json'; record = read_json(path)
        assert record['status'] == 'PASS' and record['identity']['runtime_sha256'] == source_hash()
        assert record['identity']['implementation_lock_sha256'] == implementation_sha
        proofs[stage] = digest(path)
    policy = read_json(ROOT/'configs/next2/RUNTIME_MEASUREMENT_LOCK.json')
    expected = read_json(ROOT/'artifacts/next2/latency/BL80_SHARED_V1/RESULTS.json')
    data = prepare_dev(ROOT); data.guard.admit(policy['series_ids'], purpose='Fixed packed runtime benchmark probes')
    examples = [(sid, *data.get(sid)[:2]) for sid in policy['series_ids']]
    buffer = np.concatenate([o for _, _, o in examples])[:policy['block_points']]; assert len(buffer) == 10000
    model, _ = load_artifact(ROOT/'artifacts/next2/canonical_deployment/PERFORMANCE_shared_v1')
    engines = {'FROZEN_SHARED': PreparedSharedBlendPredictor(model),
               **{v: PreparedPackedSharedPredictor(model, variant=v) for v in VARIANTS}}
    out = BASE/'engineering/packed_shared/timing'; out.mkdir(parents=True, exist_ok=True)
    order = [('FROZEN_SHARED', 'PACKED_SHARED', 'PACKED_PRECOMPUTED'),
             ('PACKED_PRECOMPUTED', 'FROZEN_SHARED', 'PACKED_SHARED'),
             ('PACKED_SHARED', 'PACKED_PRECOMPUTED', 'FROZEN_SHARED')]
    atomic_json(out/'POLICY.json', {'created_utc': utc_now(), 'hypothesis_sha256': digest(hyp),
        'runtime_sha256': source_hash(), 'implementation_lock_sha256': implementation_sha,
        'parity_proofs_sha256': proofs, 'trial_order': order,
        'timing_probe_lock_sha256': digest(ROOT/'configs/next2/RUNTIME_MEASUREMENT_LOCK.json')}, immutable=True)
    records = []; comparisons = []
    for repeat, variants in enumerate(order):
        result = {}
        for variant in variants:
            result[variant] = trial(engines[variant], variant, repeat, examples, buffer, expected, out)
            records.append(result[variant])
        a = result['FROZEN_SHARED']
        for variant in VARIANTS:
            b = result[variant]
            comparisons.append({'trial': repeat, 'variant': variant,
                'direct_10M_seconds': b['continuous_stream']['measured_prediction_seconds'],
                'direct_10M_ratio': b['continuous_stream']['measured_prediction_seconds']/a['continuous_stream']['measured_prediction_seconds'],
                'point_median_ratio': b['point_distribution']['median_us']/a['point_distribution']['median_us'],
                'H_median_ratio': b['historical_initialization']['median_us']/a['historical_initialization']['median_us'],
                'private_state_ratio': b['numeric_memory']['private_per_series_numeric_bytes']/a['numeric_memory']['private_per_series_numeric_bytes'],
                'prepared_model_ratio': b['numeric_memory']['model_and_prepared_shared_numeric_bytes']/a['numeric_memory']['model_and_prepared_shared_numeric_bytes']})
    table = pd.DataFrame(comparisons); table.to_csv(out/'PAIRED_TRIALS.csv', index=False)
    decisions = {}
    for variant in VARIANTS:
        subset = table[table.variant == variant]
        memory_gain = bool((subset.prepared_model_ratio <= .75).all() and subset.direct_10M_ratio.median() <= 1.05)
        runtime_gain = bool(subset.direct_10M_ratio.median() <= .9 and (subset.prepared_model_ratio <= 1.).all())
        common = {'median_point_within_5_percent': bool(subset.point_median_ratio.median() <= 1.05),
                  'median_H_within_10_percent': bool(subset.H_median_ratio.median() <= 1.10),
                  'private_state_within_110_percent': bool((subset.private_state_ratio <= 1.10).all()),
                  'exact_DEV_OOF_ALT_H_and_direct_10M': True, 'constant_state': True}
        decisions[variant] = {'memory_gain_route': memory_gain, 'runtime_gain_route': runtime_gain,
                              'common_gates': common, 'engineering_gate_pass': (memory_gain or runtime_gain) and all(common.values())}
    eligible = [v for v in VARIANTS if decisions[v]['engineering_gate_pass']]
    selected = eligible[0] if eligible else None
    speed_ratio = float(table[table.variant == 'PACKED_PRECOMPUTED'].direct_10M_seconds.median() /
                        table[table.variant == 'PACKED_SHARED'].direct_10M_seconds.median())
    if len(eligible) == 2 and speed_ratio <= .95: selected = 'PACKED_PRECOMPUTED'
    atomic_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(), 'variants': decisions,
        'provisional_selected_variant': selected, 'precomputed_vs_shared_median_seconds_ratio': speed_ratio,
        'total_direct_points': 90_000_000, 'new_fits': 0, 'seal_rows': 0, 'not_final_promotion': True,
        'raw_reproduction_and_public_API_still_required': True, 'paired_trials': comparisons,
        'files_sha256': {p.relative_to(out).as_posix(): digest(p) for p in out.rglob('RESULTS.json')},
        'summary_sha256': digest(out/'PAIRED_TRIALS.csv'),
        'limitation': 'Three host-conditional paired timings; OS/power not controlled. Numeric allocation sizes exclude Python/JIT objects; RSS is separately reported.'}, immutable=True)
    progress('Packed engineering benchmark complete', f'Provisional eligible variant: {selected}; public API and raw reproduction still required', experiment='E002_PACKED_SHARED')
    print('NEXT3 PACKED ENGINEERING DECISIONS', decisions, selected, flush=True)


if __name__ == '__main__': main()
