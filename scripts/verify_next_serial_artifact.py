"""Verify actual all-DEV model bytes through the API, including cold import.

These predictions are reproduction fingerprints and never quality estimates.
The original five-fold OOF checks remain the sole primary quality authority.
"""
import os
import sys
sys.dont_write_bytecode = True
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.data import prepare_dev
from src.next.candidate_io import candidate_design
from src.next.serial_inference import load_artifact, implementation_hash
from src.next.serial_variance_runtime import source_hash as runtime_hash
from scripts.verify_next_serial_contract import run_guarded
from src.utils.artifacts import write_json, sha256, utc_now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--refit-tag', default='research_v1')
    parser.add_argument('--tag', default='research_v1')
    args = parser.parse_args()
    if not all(value.replace('_', '').isalnum() for value in vars(args).values()):
        raise ValueError('Invalid artifact audit identifier')
    directory = ROOT/'artifacts/next/refits'/args.candidate/('cache_'+args.refit_tag)
    out = ROOT/'artifacts/next/all_dev_artifact_contract'/args.candidate/args.tag
    out.mkdir(parents=True, exist_ok=True)
    model, descriptor = load_artifact(directory)
    refit = json.loads((directory/'RESULTS.json').read_text(encoding='utf-8'))
    if refit['status'] != 'PASS' or descriptor['candidate']['experiment_id'] != args.candidate or refit['model_sha256'] != descriptor['model_sha256']:
        raise RuntimeError('All-DEV model or candidate identity differs')
    sources = ['scripts/verify_next_serial_artifact.py', 'scripts/check_next_serial_cold_child.py',
        'scripts/verify_next_serial_contract.py', 'next_serial_main.py', 'src/next/readonly_bootstrap.py']
    identity = {'candidate': args.candidate, 'model_directory': str(directory.relative_to(ROOT)),
        'model_sha256': sha256(directory/'model.joblib'), 'descriptor_sha256': sha256(directory/'model.json'),
        'refit_report_sha256': sha256(directory/'RESULTS.json'), 'inference_sha256': implementation_hash(),
        'runtime_sha256': runtime_hash(), 'source_sha256': {path: sha256(ROOT/path) for path in sources},
        'prediction_reference_kind': 'ALL_DEV_BATCH_FINGERPRINT_NOT_QUALITY_SCORE'}
    policy_path = out/'POLICY.json'
    if policy_path.exists():
        if json.loads(policy_path.read_text(encoding='utf-8')) != identity:
            raise RuntimeError('Artifact audit identity changed; use an explicit revision')
    else:
        write_json(policy_path, identity)
    if (out/'RESULTS.json').exists():
        result = json.loads((out/'RESULTS.json').read_text(encoding='utf-8'))
        if result['status'] != 'PASS' or result['identity'] != identity:
            raise RuntimeError('Completed artifact audit changed')
        for path, digest in result['files_sha256'].items():
            if sha256(out/path) != digest:
                raise RuntimeError('Completed artifact audit receipt changed')
        print('ALL-DEV ARTIFACT retained PASS', args.candidate, flush=True)
        return
    tick = time.perf_counter()
    data = prepare_dev(ROOT)
    design, metadata, names, feature_sources, rows = candidate_design(data, descriptor['candidate'])
    expected = np.asarray(model.predict(design), dtype=np.float32)
    if hashlib.sha256(expected.tobytes()).hexdigest() != refit['all_dev_prediction_sha256']:
        raise RuntimeError('All-DEV batch fingerprint differs from the frozen fit')
    if len(expected) != 4014405 or set(metadata['dataset_id']) != set(data.guard.dev_ids):
        raise RuntimeError('All-DEV expected coverage incomplete')
    ids = metadata['dataset_id']
    cuts = np.r_[0, np.flatnonzero(np.diff(ids))+1, len(ids)]
    slices = {int(ids[a]): (a, b) for a, b in zip(cuts[:-1], cuts[1:])}
    if len(slices) != 8000 or list(slices) != sorted(data.guard.dev_ids):
        raise RuntimeError('Expected API coordinates must be global-ID then chronological order')
    selected = json.loads((ROOT/'configs/next_latency_policy.json').read_text(encoding='utf-8'))['series_ids'][:32]
    sample = [(sid, *data.get(sid)[:2]) for sid in selected]
    active, events = [False], {'read_opens': 0, 'filesystem_mutation_attempts': 0}
    def audit(event, arguments):
        if not active[0]:
            return
        mutation = event in ('os.mkdir', 'os.remove', 'os.rename', 'os.rmdir', 'os.chmod', 'os.truncate', 'os.link', 'os.symlink')
        if event == 'open':
            _, mode, flags = arguments
            mutation = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (isinstance(flags, int) and bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
            if not mutation:
                events['read_opens'] += 1
        if mutation:
            events['filesystem_mutation_attempts'] += 1
            raise RuntimeError('All-DEV artifact inference attempted filesystem mutation')
    sys.addaudithook(audit)
    replay_path = out/'FULL_API.json'
    if not replay_path.exists():
        digest, points = hashlib.sha256(), 0
        # Bounded 1000-series batches keep memory small; each API call still
        # receives guarded outer and inner iterators and zero-read readiness.
        for start in range(0, 8000, 1000):
            batch_ids = sorted(data.guard.dev_ids)[start:start+1000]
            examples = [(h, o, len(o)) for sid in batch_ids for h, o, _ in [data.get(sid)]]
            active[0] = True
            try:
                predictions = run_guarded(examples, directory)
            finally:
                active[0] = False
            for sid, actual in zip(batch_ids, predictions):
                a, b = slices[sid]
                np.testing.assert_array_equal(actual, expected[a:b])
                digest.update(actual.tobytes())
                points += len(actual)
            print('ALL-DEV API', args.candidate, start+len(batch_ids), points, flush=True)
        if points != 4014405 or digest.hexdigest() != refit['all_dev_prediction_sha256']:
            raise RuntimeError('All-DEV full API fingerprint differs')
        write_json(replay_path, {'status': 'PASS', 'series': 8000, 'points': points,
            'model_sha256': identity['model_sha256'], 'bitwise_batch_prediction_parity': True,
            'output_sha256': digest.hexdigest(), 'inference_audit': dict(events),
            'single_pass_and_zero_read_readiness': True, 'quality_score_computed': False})
    replay = json.loads(replay_path.read_text(encoding='utf-8'))
    if replay['status'] != 'PASS' or replay['model_sha256'] != identity['model_sha256'] or replay['output_sha256'] != refit['all_dev_prediction_sha256']:
        raise RuntimeError('Retained full API fingerprint changed')
    suffix_path = out/'SUFFIX_CONTRACT.json'
    if not suffix_path.exists():
        active[0] = True
        try:
            examples = [(h, o, len(o)) for sid, h, o in sample]
            first, repeated = run_guarded(examples, directory), run_guarded(examples, directory)
            suffix_cases = 0
            for (sid, h, o), left, right in zip(sample, first, repeated):
                a, b = slices[sid]
                np.testing.assert_array_equal(left, right)
                np.testing.assert_array_equal(left, expected[a:b])
                for prefix in sorted({min(length, len(o)) for length in (1, 8, 64, 257)}):
                    for suffix in (np.full(5, -1e5, dtype=np.float32), np.full(17, np.nan, dtype=np.float32), np.linspace(-500, 500, 257, dtype=np.float32)):
                        actual = run_guarded([(h, np.r_[o[:prefix], suffix], prefix)], directory, prefix_only=True)[0]
                        np.testing.assert_array_equal(actual, left[:prefix])
                        suffix_cases += 1
            assert len(run_guarded([(sample[0][1], np.empty(0, dtype=np.float32), 0)], directory)[0]) == 0
        finally:
            active[0] = False
        write_json(suffix_path, {'status': 'PASS', 'model_sha256': identity['model_sha256'], 'series': 32,
            'modified_future_suffix_cases': suffix_cases, 'repeat_predictions_bitwise_equal': True,
            'single_pass_and_no_read_ahead': True, 'zero_read_readiness': True, 'empty_online_no_prediction': True,
            'future_nan_suffix_not_consumed': True, 'inference_audit': dict(events), 'quality_score_computed': False})
    cold_path = out/'COLD_IMPORT.json'
    if not cold_path.exists():
        fixture_path = ROOT/'data/next/all_dev_artifact_contract'/args.candidate/args.tag/'fixture.json'
        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        series = []
        for sid, h, o in sample:
            a, b = slices[sid]
            series.append({'historical': h.tolist(), 'online': o.tolist(), 'expected': expected[a:b].tolist()})
        write_json(fixture_path, {'model_directory': str(directory), 'series': series,
            'model_sha256': identity['model_sha256'], 'prediction_reference_kind': identity['prediction_reference_kind']})
        subprocess.run([sys.executable, str(ROOT/'scripts/check_next_serial_cold_child.py'),
            '--fixture', str(fixture_path), '--output', str(cold_path)], cwd=ROOT, check=True)
    cold, suffix = json.loads(cold_path.read_text(encoding='utf-8')), json.loads(suffix_path.read_text(encoding='utf-8'))
    for report in (cold, suffix):
        if report['status'] != 'PASS' or report['model_sha256'] != identity['model_sha256']:
            raise RuntimeError('Artifact contract receipt mismatch')
    if cold['audit']['filesystem_mutation_attempts'] or suffix['inference_audit']['filesystem_mutation_attempts']:
        raise RuntimeError('Artifact contract attempted filesystem mutation')
    write_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(), 'identity': identity,
        'all_dev_full_api_points': 4014405, 'all_dev_full_api_bitwise_equal': True,
        'cold_import_readonly': True, 'single_pass_and_future_suffix_contract': True,
        'files_sha256': {path.name: sha256(path) for path in (replay_path, suffix_path, cold_path)},
        'feature_sources': feature_sources, 'elapsed_seconds': time.perf_counter()-tick,
        'quality_score_computed': False, 'seal_rows': 0, 'reduced_new_uses': 0,
        'scope': 'Actual complete-DEV artifact, all DEV observations through guarded infer, repeated/altered-suffix contracts, and fresh-process Python audit write denial before imports. Native syscalls and cloud execution are not claimed.'})
    print('ALL-DEV ARTIFACT COMPLETE', args.candidate, flush=True)


if __name__ == '__main__':
    main()
