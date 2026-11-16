"""Full pinned official run/execute/check with generated data and NEXT inference.

Authentication, SDK .test(), actual reduced data, and cloud submission are never
used. The official literal reduced filename contains newly generated values.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import contextlib
import io
import json
import os
import subprocess
import time
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.validation.protocol import official_runner
from src.validation.synthetic import fixture, KINDS
from src.data.labels import labels_from_tau
from src.next.guard import SealGuard
from src.utils.artifacts import write_json, sha256, utc_now
from scripts.verify_scorer import load_official


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', default='Z001_LOG_ENERGY_CURRENT')
    parser.add_argument('--fixture-tag', default='v2')
    parser.add_argument('--model-directory')
    parser.add_argument('--tag', default='research_v1')
    args = parser.parse_args()
    if not args.tag.replace('_', '').isalnum():
        raise ValueError('Simple explicit runner tag required')
    guard = SealGuard(ROOT)
    source = ROOT/args.model_directory if args.model_directory else ROOT/'artifacts/next/deployment/contracts'/args.candidate/args.fixture_tag
    descriptor = json.loads((source/'model.json').read_text(encoding='utf-8'))
    if descriptor['candidate']['experiment_id'] != args.candidate or sha256(source/'model.joblib') != descriptor['model_sha256']:
        raise RuntimeError('Requested NEXT runner artifact changed')
    out = ROOT/'artifacts/next/official_runner'/args.candidate/args.tag
    if out.exists():
        raise RuntimeError('Keep the completed runner evidence; use a new tag for a changed artifact')
    out.mkdir(parents=True)
    base = ROOT/'data/next/official_protocol_fixture'/args.candidate/args.tag
    data, model, prediction = base/'data', base/'model', base/'prediction'
    for directory in (data, model, prediction):
        directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    frames, targets, indices, series = [], [], [], []
    for sid in range(20):
        historical, online, tau = fixture(KINDS[sid%len(KINDS)], seed=20260908+sid,
            historical_length=1000+sid*3, online_length=120+sid%3*7, break_at=40)
        start = 100+7*sid
        keys = pd.MultiIndex.from_product([[sid], range(start, start+len(historical)+len(online))], names=['id', 'time'])
        frames.append(pd.DataFrame({'value': np.r_[historical, online],
            'period': np.r_[np.ones(len(historical), dtype=np.int8), np.full(len(online), 2, dtype=np.int8)]}, index=keys))
        targets.append(pd.DataFrame({'target': labels_from_tau(len(online), tau)}, index=keys[len(historical):]))
        indices.append({'id': sid, 'tau_index': -1 if tau is None else tau})
        series.append((historical, online))
    x, y = pd.concat(frames), pd.concat(targets)
    x.to_parquet(data/'X_train.parquet')
    x.to_parquet(data/'X_test.reduced.parquet')
    y.to_parquet(data/'y_test.parquet')
    pd.DataFrame(indices).set_index('id').to_parquet(data/'y_train_index.parquet')
    calls, logs, determinism = [], [], []
    class Context:
        is_local = True
        force_first_train = True
        is_determinism_check_enabled = True
        def execute(self, command, parameters=None, **kwargs):
            env = os.environ.copy()
            for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
                env[name] = '1'
            result = subprocess.run([sys.executable, str(ROOT/'scripts/next_registered_runner_worker.py'),
                '--command', command, '--data', str(data), '--model', str(model), '--source-model', str(source),
                '--parameters', json.dumps(parameters or {})], cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
            calls.append({'command': command, 'exit_code': result.returncode})
            logs.append(result.stdout+result.stderr)
            if result.returncode:
                raise RuntimeError(result.stderr)
        def log(self, message, **kwargs):
            logs.append(str(message))
        def report_determinism(self, value):
            determinism.append(bool(value))
    class Tracer:
        def span(self, *args, **kwargs):
            return contextlib.nullcontext()
    official_runner().run(Context(), Tracer(), str(data), str(prediction))
    with contextlib.redirect_stdout(io.StringIO()):
        load_official().check(str(prediction), str(data))
    frame = pd.read_parquet(prediction/'prediction.parquet')
    import next_registered_main as entry
    direct = np.asarray(list(entry.infer(iter(series), model))[1:], dtype=np.float32)
    np.testing.assert_array_equal(frame.prediction.to_numpy(), direct)
    if len(frame) != len(y) or not frame.index.equals(y.index) or determinism != [True]:
        raise RuntimeError('Official wire/index/determinism contract failed')
    guard.check_state()
    (out/'runner.log').write_text('\n'.join(logs), encoding='utf-8')
    report = {'status': 'PASS', 'created_utc': utc_now(), 'candidate': args.candidate,
        'model_sha256': sha256(source/'model.joblib'), 'descriptor_sha256': sha256(source/'model.json'),
        'entrypoint': 'next_registered_main.py', 'entry_sha256': sha256(ROOT/'next_registered_main.py'),
        'official_run_function': True, 'official_execute_function': True, 'official_prediction_check': True,
        'subprocess_calls': calls, 'rows': len(frame), 'synthetic_series': len(series), 'prediction_dtype': str(frame.prediction.dtype),
        'index_names': frame.index.names, 'indices_match': True, 'nonzero_and_series_varying_time_origins': True,
        'official_10_percent_determinism': determinism, 'direct_vs_wire_max_abs_diff': 0.,
        'training_function': 'Only this synthetic protocol fixture stages an already fitted artifact after consuming 20 generated training tuples. Complete 8000-DEV train/refit is tested separately; no unknown synthetic ID is admitted to the production trainer.',
        'input_filename_note': 'X_test.reduced.parquet is an unchanged official-runner literal containing only values generated by this script. No actual reduced dataset is opened or scored.',
        'authenticated_sdk_test_called': False, 'cloud_sandbox_validated': False, 'competition_evaluation_data_used': False,
        'seal_rows': 0, 'reduced_new_uses': 0, 'public_submissions': 0, 'elapsed_seconds': time.perf_counter()-started,
        'source_sha256': {name: sha256(ROOT/name) for name in ['vendor/official/runner.py', 'vendor/official/scoring.py',
            'scripts/next_registered_runner_worker.py', 'scripts/verify_next_registered_runner.py']},
        'log_sha256': sha256(out/'runner.log')}
    write_json(out/'RESULTS.json', report)
    print('NEXT FULL OFFICIAL LOCAL RUNNER PASS', args.candidate, len(frame), flush=True)


if __name__ == '__main__':
    main()
