"""Unchanged selected BL80/D2 entries against batch features on RP only.

The retained weights historically used full DEV. This is inference engineering
verification, not held-out quality evidence, a fresh fit, or new role selection.
"""
import os
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import hashlib
import importlib
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.next5.artifacts import BASE, atomic_json, read_json, digest, utc_now, progress
from src.next5.data import prepare_research
from src.next5.stream_contracts import Datasets
from src.next5.scheduler import register_job


def matrix(model, historical, online):
    state = model.make_state(historical)
    result = np.empty((len(online), len(model.names)), dtype=np.float32)
    for i, value in enumerate(online):
        result[i] = state.update(float(value))
    return result


def collect(entry, directory, historical, online):
    source = Datasets([(0, historical, online)])
    stream = entry.infer(source, directory)
    assert source.accessed == 0 and next(stream) is None and source.accessed == 0
    values = np.empty(len(online), dtype=np.float64)
    for i in range(len(online)):
        values[i] = next(stream)
        assert source.current.consumed == i + 1
    try:
        next(stream)
    except StopIteration:
        pass
    else:
        raise AssertionError('Extra retained-entry output')
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--register', action='store_true')
    args = parser.parse_args()
    parent = BASE / 'retained_artifact_validation'
    refs = read_json(BASE / 'preflight/FROZEN_REFERENCES.json')
    if args.register:
        roles = {name: refs['roles'][name] for name in ('FROZEN_PRIMARY', 'NEXT4_ROBUSTNESS')}
        sources = ('scripts/verify_next5_retained_artifacts.py', 'src/next5/stream_contracts.py',
            'src/next5/data.py', 'src/next5/guard.py', 'src/next2/shared_inference.py',
            'src/next4/robust_finalist.py', 'src/next2/blend.py',
            'next2_shared_performance_main.py', 'next4_robustness_main.py')
        atomic_json(parent / 'POLICY_LOCK.json', {'created_utc': utc_now(), 'roles': roles,
            'sources_sha256': {n: digest(ROOT / n) for n in sources},
            'research_lock_sha256': digest(ROOT / 'NEXT5_SHADOW_LOCK.json'),
            'expected_series': 6400, 'expected_points': 3203157, 'new_fits': 0,
            'prefix_cases_per_role': 100, 'prefix_seed': 20260908,
            'real_scope': 'ResearchPool only; all real points through actual retained public entries',
            'oracle': 'Original complete feature states, batch compact-tree predictions and original float32 combine; not the shared scalar runtime',
            'labels_supplied_to_inference': False, 'quality_score_computed': False,
            'historical_training_scope': 'Old weights used all DEV, including historical Shadow; this audit is not independent quality validation',
            'reset_check': 'First 20 ResearchPool series replayed in reverse order',
        }, immutable=True)
        register_job({'id': 'RETAINED_ARTIFACT_RP_API_PARITY', 'kind': 'validation', 'priority': 87,
            'depends_on_success': ['AUDIT_PRIMITIVE_FIT_INVENTORY'],
            'argv': ['scripts/verify_next5_retained_artifacts.py'],
            'description': 'Original selected BL80/D2 public entries versus original batch feature/tree predictions on all 6400 RP series; no quality score or fit',
            'retry_safe': False})
        progress('Retained artifact RP inference audit registered', 'Unchanged actual entries, independent batch-feature oracle, no new fit and no historical Shadow reads')
        return
    policy = read_json(parent / 'POLICY_LOCK.json')
    for name, sha in policy['sources_sha256'].items():
        assert digest(ROOT / name) == sha
    assert digest(ROOT / 'NEXT5_SHADOW_LOCK.json') == policy['research_lock_sha256']
    out = parent / 'research_v1'
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / 'START.json', {'created_utc': utc_now(), 'policy_sha256': digest(parent / 'POLICY_LOCK.json'),
        'new_fits': 0, 'quality_score_computed': False}, immutable=True)
    data = prepare_research()
    active = [False]
    attempts = []
    def audit(event, arguments):
        if not active[0]:
            return
        write = False
        if event == 'open':
            path, mode, flags = arguments
            write = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            if write and isinstance(path, (str, bytes, os.PathLike)) and os.path.normcase(os.path.normpath(os.fsdecode(path))) == os.path.normcase(os.path.normpath(os.devnull)):
                return
        if event in ('os.mkdir', 'os.remove', 'os.rmdir', 'os.rename', 'os.chmod', 'os.truncate', 'os.link', 'os.symlink'):
            write = True
        if write:
            attempts.append(event)
            raise RuntimeError('Retained artifact audit observed a persistent write: ' + event)
    sys.addaudithook(audit)
    started = time.perf_counter()
    role_names = ('FROZEN_PRIMARY', 'NEXT4_ROBUSTNESS')
    try:
        active[0] = True
        entries = {name: importlib.import_module(Path(policy['roles'][name]['entry']).stem) for name in role_names}
        directories = {name: (ROOT / policy['roles'][name]['model_file']).parent for name in role_names}
        for name in role_names:
            assert digest(ROOT / policy['roles'][name]['model_file']) == policy['roles'][name]['model_sha256']
        from src.next2.shared_inference import load_artifact
        from src.next4.robust_finalist import load_prepared
        from src.next2.blend import combine
        baseline, _ = load_artifact(directories['FROZEN_PRIMARY'])
        expert = load_prepared(directories['NEXT4_ROBUSTNESS']).model
        def batch(h, o):
            left = baseline.primary.predict(matrix(baseline.primary, h, o))
            right = baseline.complement.predict(matrix(baseline.complement, h, o))
            base = combine(left, right, .8)
            extra = expert.predict(matrix(expert, h, o))
            return {'FROZEN_PRIMARY': base.astype(np.float64),
                    'NEXT4_ROBUSTNESS': combine(base, extra, .9).astype(np.float64)}
        rng = np.random.default_rng(policy['prefix_seed'])
        for case in range(policy['prefix_cases_per_role']):
            h = rng.standard_t(5, [192, 256, 512, 1024][case % 4]).astype(np.float32)
            o = rng.normal(size=72).astype(np.float32)
            cutoff = case % 64 + 1
            changed = np.r_[o, rng.normal(size=17).astype(np.float32)]
            changed[cutoff:] = rng.normal(0, 20, len(changed) - cutoff).astype(np.float32)
            expected = batch(h, o)
            for name in role_names:
                full = collect(entries[name], directories[name], h, o)
                altered = collect(entries[name], directories[name], h, changed)
                prefix = collect(entries[name], directories[name], h, o[:cutoff])
                np.testing.assert_array_equal(full, expected[name])
                np.testing.assert_array_equal(full[:cutoff], altered[:cutoff])
                np.testing.assert_array_equal(full[:cutoff], prefix)
            if case % 20 == 19:
                print('NEXT5 RETAINED PREFIX ORACLE', case + 1, flush=True)
        def records(indices):
            for j in indices:
                sid = int(data.ids[j])
                h, o, _ = data.get(sid)
                yield sid, h, o
        sources = {name: Datasets(records(range(len(data.ids)))) for name in role_names}
        streams = {name: entries[name].infer(sources[name], directories[name]) for name in role_names}
        for name in role_names:
            assert next(streams[name]) is None and sources[name].accessed == 0
        hashes = {name: hashlib.sha256() for name in role_names}
        retained = {name: {} for name in role_names}
        count = 0
        for j, sid in enumerate(data.ids):
            h, o, _ = data.get(int(sid))
            expected = batch(h, o)
            for name in role_names:
                values = np.empty(len(o), dtype=np.float64)
                for k in range(len(o)):
                    values[k] = next(streams[name])
                    assert sources[name].sid == int(sid) and sources[name].current.consumed == k + 1
                assert np.isfinite(values).all() and np.all((values >= 0) & (values <= 1))
                np.testing.assert_array_equal(values, expected[name])
                hashes[name].update(values.tobytes())
                if j < 20:
                    retained[name][j] = values.copy()
            count += len(o)
            if j % 128 == 127:
                print('NEXT5 RETAINED REAL RP ORACLE', j + 1, count, flush=True)
        for name in role_names:
            try:
                next(streams[name])
            except StopIteration:
                pass
            else:
                raise AssertionError('Extra retained entry output after RP exhaustion')
            reverse = Datasets(records(reversed(range(20))))
            stream = entries[name].infer(reverse, directories[name])
            assert next(stream) is None and reverse.accessed == 0
            for j in reversed(range(20)):
                values = np.asarray([next(stream) for _ in range(len(retained[name][j]))], dtype=np.float64)
                np.testing.assert_array_equal(values, retained[name][j])
            try:
                next(stream)
            except StopIteration:
                pass
            else:
                raise AssertionError('Extra reverse-order output')
        assert len(data.ids) == policy['expected_series'] and count == policy['expected_points']
        assert not attempts
    finally:
        active[0] = False
    data.guard.check_state()
    atomic_json(out / 'RESULTS.json', {'created_utc': utc_now(), 'status': 'PASS', 'series_per_role': len(data.ids),
        'real_points_per_role': count, 'roles': {name: {'model_sha256': policy['roles'][name]['model_sha256'],
            'RP_output_float64_sha256': hashes[name].hexdigest(), 'batch_and_actual_API_bitwise_equal': True,
            'prefix_cases': 100, 'reverse_reset_series': 20} for name in role_names},
        'first_none_before_any_data': True, 'one_current_observation_per_output': True,
        'future_length_and_suffix_traps': True, 'persistent_write_attempts': len(attempts),
        'batch_oracle_uses_original_complete_features': True, 'old_per_series_predictions_loaded': False,
        'quality_score_computed': False, 'new_fits': 0, 'Shadow_rows': 0, 'LOCAL_SEAL_rows': 0,
        'historical_full_DEV_training_does_not_make_this_heldout_quality_evidence': True,
        'elapsed_seconds': time.perf_counter() - started, 'policy_sha256': digest(parent / 'POLICY_LOCK.json'),
        'source_sha256': digest(Path(__file__))}, immutable=True)
    progress('Retained selected entries verified on RP', 'BL80 and D2 exact batch/API parity for all 6400 series; no quality score, fit or Shadow access')


if __name__ == '__main__':
    main()
