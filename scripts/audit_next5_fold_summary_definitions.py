"""Disambiguate frozen fold summaries using existing scalar receipts only."""
from pathlib import Path
from statistics import mean, median
import csv
import math
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next5.artifacts import BASE, atomic_json, read_json, digest, utc_now, progress


def main():
    policy_path = ROOT / 'configs/next5/RESEARCH_POLICY_LOCK.json'
    gate = read_json(policy_path)['performance_gate']
    inputs = {policy_path.relative_to(ROOT).as_posix(): digest(policy_path)}
    rows = []
    for path in sorted((BASE / 'experiments').glob('*/full/RESULTS.json')):
        r = read_json(path)
        inputs[path.relative_to(ROOT).as_posix()] = digest(path)
        new = [f['TS_AUC'] for f in r['fold_results']]
        old = [f['BL80_TS_AUC'] for f in r['fold_results']]
        delta = [a - b for a, b in zip(new, old, strict=True)]
        assert len(delta) == 5
        expected = {'mean_delta': mean(delta), 'median_delta': median(new) - median(old),
            'worst_delta': min(new) - min(old), 'minimum_fold_delta': min(delta)}
        for key, value in expected.items():
            assert math.isclose(value, r[key], rel_tol=0., abs_tol=1e-15), (r['candidate'], key)
        boot_path = BASE / 'bootstrap' / r['candidate'] / 'RESULTS.json'
        boot = read_json(boot_path)['paired_all_time']
        inputs[boot_path.relative_to(ROOT).as_posix()] = digest(boot_path)
        assert boot['estimate_median_of_folds_delta'] == r['median_delta']
        rows.append({'candidate': r['candidate'], **expected,
            'median_matched_fold_delta': median(delta),
            'original_median_floor_pass': r['median_delta'] >= gate['median_delta_min'],
            'matched_median_floor_pass': median(delta) >= gate['median_delta_min'],
            'original_worst_floor_pass': r['worst_delta'] >= gate['worst_delta_min'],
            'matched_worst_floor_pass': min(delta) >= gate['worst_delta_min'],
            'mean_floor_pass': r['mean_delta'] >= gate['mean_delta_min']})
    assert len(rows) == 19 and not any(r['mean_floor_pass'] for r in rows)
    median_changed = [r['candidate'] for r in rows if r['original_median_floor_pass'] != r['matched_median_floor_pass']]
    worst_changed = [r['candidate'] for r in rows if r['original_worst_floor_pass'] != r['matched_worst_floor_pass']]
    out = BASE / 'integrity_audits/fold_summary_definitions_v1'
    out.mkdir(parents=True, exist_ok=False)
    table_path = out / 'FOLD_SUMMARY_DEFINITIONS.csv'
    with table_path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(out / 'RESULTS.json', {'created_utc': utc_now(), 'status': 'PASS',
        'full_candidates': 19, 'source_statistics_preserved': True,
        'candidates_with_changed_median_floor_result': median_changed,
        'candidates_with_changed_worst_floor_result': worst_changed,
        'all_full_candidates_fail_original_mean_floor': True,
        'selection_effect': 'No candidate eligible under either interpretation',
        'bootstrap_median_interval_target': 'Difference of score medians; no interval claimed for median matched difference',
        'new_fits': 0, 'new_predictions': 0, 'new_draws': 0, 'real_point_arrays_read': False,
        'inputs_sha256': inputs, 'table_sha256': digest(table_path),
        'source_sha256': digest(Path(__file__))}, immutable=True)
    progress('Fold summary targets disambiguated', 'Existing 19 full scalar receipts; original gates and CI targets preserved')
    print('NEXT5 FOLD DEFINITIONS PASS', median_changed, worst_changed, flush=True)


if __name__ == '__main__':
    main()
