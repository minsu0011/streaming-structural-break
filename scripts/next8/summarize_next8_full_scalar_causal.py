"""Verify complete scalar census coverage and retain every case in a CSV."""
from pathlib import Path
import csv
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.next8
from src.next8.common import BASE, read_json, digest, atomic, utc_now, activity


def main():
    with activity('Verify and publish every actual scalar census case', 'FINAL_AUDIT', 'B7_NORMALIZATION_AND_LINEAGE'):
        authorities = {}
        for path in (BASE / 'full_replay_audit').glob('SHARD_*.json'):
            receipt = read_json(path)
            assert receipt['status'] == 'PASS'
            for record in receipt['results']:
                sid = record['series_id']
                assert sid not in authorities
                authorities[sid] = (record['outer_fold'], record['online_points'])
        assert len(authorities) == 6400
        seen = set()
        shard_manifest = {}
        totals = {'series': 0, 'online_points': 0, 'paths': 0, 'scalar_prediction_points': 0}
        errors = {}
        table = BASE / 'FULL_SCALAR_CAUSAL_CASES_NEXT8.csv'
        with table.open('x', encoding='utf-8-sig', newline='') as stream:
            writer = None
            for part in (0, 1):
                part_result = read_json(BASE / f'FULL_SCALAR_CAUSAL_PART_{part}_NEXT8.json')
                assert part_result['status'] == 'PASS' and part_result['series'] == 3200
                for shard in part_result['shards']:
                    path = BASE / shard['path']
                    assert digest(path) == shard['sha256']
                    receipt = read_json(path)
                    assert receipt['status'] == 'PASS'
                    assert receipt['policy_sha256'] == digest(BASE / 'FULL_SCALAR_CAUSAL_AUDIT_POLICY_NEXT8.json')
                    assert receipt['implementation_lock_sha256'] == digest(BASE / 'FULL_SCALAR_CAUSAL_AUDIT_LOCK_NEXT8.json')
                    shard_manifest[shard['path']] = shard['sha256']
                    for record in receipt['records']:
                        sid = record['series_id']
                        assert sid not in seen and authorities[sid] == (record['fold'], record['online_points'])
                        seen.add(sid)
                        totals['series'] += 1
                        totals['online_points'] += record['online_points']
                        assert set(record['modes']) == {'NORMAL', 'EARLY'}
                        for mode, checks in record['modes'].items():
                            assert set(checks) == {'M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6'}
                            for method, check in checks.items():
                                assert check['points'] == record['online_points']
                                assert all(check[k] for k in ('fresh_scalar_chunk_bitwise', 'pickle_resume_at73_bitwise',
                                    'future_suffix_prefix73_bitwise', 'only_H_and_current_scalar_given_to_state'))
                                error = check['maximum_saved_absolute_error']
                                assert error is None or error <= check['saved_tolerance']
                                key = mode + '_' + method
                                errors[key] = max(errors.get(key, 0.), error or 0.)
                                row = {'series_id': sid, 'fold': record['fold'], 'H_length': record['H_length'],
                                       'mode': mode, 'method': method, **check}
                                if writer is None:
                                    writer = csv.DictWriter(stream, fieldnames=list(row))
                                    writer.writeheader()
                                writer.writerow(row)
                                totals['paths'] += 1
                                totals['scalar_prediction_points'] += check['points']
        assert seen == set(authorities) and totals['online_points'] == 3203157
        assert totals['paths'] == 89600 and totals['scalar_prediction_points'] == 44844198
        result = {'status': 'PASS', 'timestamp': utc_now(), **totals,
            'actual_scalar_pickle_chunk_future_prefix_complete_RP_census': True,
            'original_held_out_fold_assignment_verified': True, 'methods': ['M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6'],
            'modes': ['NORMAL', 'EARLY'], 'maximum_saved_absolute_error_by_mode_method': errors,
            'case_table': table.name, 'case_table_sha256': digest(table), 'case_table_rows': totals['paths'],
            'case_table_retains_each_original_prediction_hash_and_check': True,
            'shard_manifest_sha256': shard_manifest,
            'part_receipt_sha256': {str(p): digest(BASE / f'FULL_SCALAR_CAUSAL_PART_{p}_NEXT8.json') for p in (0, 1)},
            'policy_sha256': digest(BASE / 'FULL_SCALAR_CAUSAL_AUDIT_POLICY_NEXT8.json'),
            'implementation_lock_sha256': digest(BASE / 'FULL_SCALAR_CAUSAL_AUDIT_LOCK_NEXT8.json'),
            'new_fits_quality_scores_or_latency_promotion_decisions': 0,
            'limits': ['Original scalar prediction count is reported once per case; replay variants are not independent evidence.',
                       'Aliases and M6 fallback are actual named APIs, not independent fitted models.',
                       'Normal and early reuse the same6400 groups, not12800 independent groups.',
                       'Saved comparison is only on frozen scored rows; early warmup remains label-free.',
                       'Recorded paths and fixed transformations do not prove correctness for every possible future input.']}
        atomic(BASE / 'FULL_SCALAR_CAUSAL_AUDIT_NEXT8.json', result)


if __name__ == '__main__':
    main()
