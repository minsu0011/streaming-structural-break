"""Census of actual held-out scalar APIs, using the frozen portable checks.

This is a final execution audit. No label, quality metric, refit or selector
is added. Every RP series uses its original excluded fold in both modes.
"""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.next8
import numpy as np
from src.next8.common import BASE, atomic, read_json, digest, utc_now, activity
from src.next8.cache import FeatureCache
from src.next8.data import outer_scope
from src.next8.nested_auxiliary import early_scope
from src.next8.runtime_audit import load_scope_runtime, observations
from src.next8.streaming_runtime import IntegratedState
from src.next8.portable_replay import run
from src.next8.integration import fixed_blend


def main(part):
    policy = read_json(BASE / 'FULL_SCALAR_CAUSAL_AUDIT_POLICY_NEXT8.json')
    assert policy['status'] == 'FROZEN'
    lock = read_json(BASE / 'FULL_SCALAR_CAUSAL_AUDIT_LOCK_NEXT8.json')
    for name, sha in lock['source_sha256'].items():
        assert digest(ROOT / name) == sha, name
    admission = read_json(BASE / 'runtime/FINAL_AUDIT_ADMISSION.json')
    assert admission['status'] == 'PASS'
    assert digest(BASE / 'PREDICTION_FREEZE_NEXT8.json') == policy['prediction_freeze_sha256']
    out = BASE / 'full_scalar_causal_audit'
    out.mkdir(parents=True, exist_ok=True)
    with activity(f'Full scalar API census load part{part}', 'FINAL_AUDIT', 'B7_NORMALIZATION_AND_LINEAGE'):
        cache = FeatureCache(verify_values=True)
        assert len(cache.data.ids) == 6400
        bundles = {}
        for early in (False, True):
            for fold in range(5):
                scope = early_scope(outer_scope(fold, cache.data), early)
                bundles[fold, early] = load_scope_runtime(cache, scope, fold, early=early)
    shards = []
    for shard, start in enumerate(range(0, 6400, 32)):
        if shard % 2 != part:
            continue
        target = out / f'SHARD_{shard:03d}.json'
        if target.exists():
            receipt = read_json(target)
            assert receipt['status'] == 'PASS' and receipt['policy_sha256'] == digest(BASE / 'FULL_SCALAR_CAUSAL_AUDIT_POLICY_NEXT8.json')
            assert receipt['implementation_lock_sha256'] == digest(BASE / 'FULL_SCALAR_CAUSAL_AUDIT_LOCK_NEXT8.json')
        else:
            with activity(f'All seven normal and early scalar APIs, RP shard{shard}', 'FINAL_AUDIT', 'B7_NORMALIZATION_AND_LINEAGE'):
                records = []
                for sid in cache.data.ids[start:start + 32]:
                    sid = int(sid)
                    historical, online, rows = observations(cache, sid)
                    fold = cache.data.guard.split.assignment(sid)[1]
                    modes = {}
                    for early in (False, True):
                        heads, baseline, reference, saved, states, integration = bundles[fold, early]
                        factories = {'M0': baseline.new_state, **{name: head.new_state for name, head in heads.items()},
                                     'M6': lambda h: IntegratedState(baseline, heads['M5'], h, integration['eligible'])}
                        assert set(factories) == {'M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6'}
                        checks = {}
                        for method, factory in factories.items():
                            values, check = run(factory, historical, online)
                            if method in ('M0', 'M6'):
                                keep = np.isin(rows, reference['global_row'])
                                at = np.searchsorted(reference['global_row'], rows[keep])
                                np.testing.assert_array_equal(reference['global_row'][at], rows[keep])
                                expected = reference['prediction'][at]
                                if method == 'M6' and integration['eligible']:
                                    hat = np.searchsorted(saved['global_row'], rows[keep])
                                    np.testing.assert_array_equal(saved['global_row'][hat], rows[keep])
                                    expected = fixed_blend(expected, saved['M5'][hat])
                                tolerance = 1e-12 if method == 'M0' else 1e-12 + .05 * np.finfo(np.float32).eps
                            else:
                                keep = np.isin(rows, saved['global_row'])
                                at = np.searchsorted(saved['global_row'], rows[keep])
                                np.testing.assert_array_equal(saved['global_row'][at], rows[keep])
                                expected = saved[method][at]
                                tolerance = np.finfo(np.float32).eps
                            error = float(np.max(np.abs(values[keep] - expected))) if keep.any() else None
                            assert error is None or error <= tolerance, (sid, early, method, error)
                            checks[method] = {**check, 'saved_comparison_points': int(keep.sum()),
                                'maximum_saved_absolute_error': error, 'saved_tolerance': float(tolerance),
                                'actual_alias_method': heads[method].method if method in heads else method,
                                'M6_eligible': integration['eligible'] if method == 'M6' else None}
                        modes['EARLY' if early else 'NORMAL'] = checks
                    records.append({'series_id': sid, 'fold': int(fold), 'H_length': len(historical),
                                    'online_points': len(online), 'modes': modes})
                receipt = {'status': 'PASS', 'timestamp': utc_now(), 'part': part, 'shard': shard,
                    'series': len(records), 'online_points': sum(r['online_points'] for r in records),
                    'records': records, 'policy_sha256': digest(BASE / 'FULL_SCALAR_CAUSAL_AUDIT_POLICY_NEXT8.json'),
                    'implementation_lock_sha256': digest(BASE / 'FULL_SCALAR_CAUSAL_AUDIT_LOCK_NEXT8.json'),
                    'admission_sha256': digest(BASE / 'runtime/FINAL_AUDIT_ADMISSION.json'),
                    'new_model_fits_or_quality_scores': 0, 'break_labels_tau_or_future_as_predictor_input': False}
                atomic(target, receipt)
        shards.append({'shard': shard, 'path': target.relative_to(BASE).as_posix(), 'sha256': digest(target),
                       'series': receipt['series'], 'online_points': receipt['online_points']})
    assert len(shards) == 100 and sum(r['series'] for r in shards) == 3200
    atomic(BASE / f'FULL_SCALAR_CAUSAL_PART_{part}_NEXT8.json', {'status': 'PASS', 'timestamp': utc_now(),
        'part': part, 'shards': shards, 'series': 3200, 'online_points': sum(r['online_points'] for r in shards),
        'methods_per_series': 14, 'all_actual_scalar_pickle_chunk_future_prefix_checks_passed': True,
        'no_new_fit_quality_score_or_latency_promotion_claim': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=int, choices=(0, 1), required=True)
    main(parser.parse_args().part)
