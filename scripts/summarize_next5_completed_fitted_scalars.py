"""Descriptive review of completed fixed artifacts; no fitting or resampling."""
from pathlib import Path
import argparse
import csv
import sys
from collections import defaultdict
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next5.artifacts import BASE, read_json, atomic_json, digest, utc_now, progress


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', required=True)
    args = parser.parse_args()
    available = {p.parent.parent.name for p in (BASE / 'experiments').glob('*/full/RESULTS.json')}
    assert args.candidate in available, 'An existing completed full candidate is required'
    grouped, inputs = defaultdict(list), {}
    for fold in range(5):
        directory = BASE / 'synthetic/candidates' / args.candidate / 'full' / f'fold_{fold}' / 'ANALYSIS'
        receipt_path = directory / 'RESULTS.json'
        receipt = read_json(receipt_path)
        assert receipt['status'] == 'PASS' and receipt['fold'] == fold and receipt['candidate'] == args.candidate
        inputs[receipt_path.relative_to(ROOT).as_posix()] = digest(receipt_path)
        tables = {}
        for name in ('PAIRED_COUNTEREXAMPLES.csv', 'PAIRED_OUTLIER_RESPONSE.csv', 'LONG_NULL_RESULTS.csv'):
            path = directory / name
            assert digest(path) == receipt['files_sha256'][name]
            inputs[path.relative_to(ROOT).as_posix()] = digest(path)
            with path.open(encoding='utf-8-sig', newline='') as stream:
                tables[name] = list(csv.DictReader(stream))
        for row in tables['PAIRED_COUNTEREXAMPLES.csv']:
            if row['reference'] == 'BL80':
                grouped[('TS_AUC_delta', row['scenario'])].append({'fold': fold,
                    'delta': float(row['estimate_mean_delta']),
                    'conditional_seed_ci_2_5': float(row['ci_2_5']),
                    'conditional_seed_ci_97_5': float(row['ci_97_5'])})
        for row in tables['PAIRED_OUTLIER_RESPONSE.csv']:
            if row['reference'] == 'BL80' and row['metric'] == 'mean_excess_at_horizon1024':
                grouped[('impulse_mean_excess_1024_delta', row['scenario'])].append({'fold': fold,
                    'delta': float(row['estimate_mean_delta']),
                    'conditional_seed_ci_2_5': float(row['ci_2_5']),
                    'conditional_seed_ci_97_5': float(row['ci_97_5'])})
        for scenario in ('N0', 'N1', 'N2', 'N3'):
            values = {row['model']: row for row in tables['LONG_NULL_RESULTS.csv']
                if row['scenario'] == scenario and int(row['horizon']) == 16384 and float(row['alpha']) == .05}
            old = float(values['BL80']['null_eventual_alarm_rate'])
            new = float(values['candidate']['null_eventual_alarm_rate'])
            grouped[('eventual_alarm_rate_16384_alpha005_delta', scenario)].append(
                {'fold': fold, 'candidate': new, 'BL80': old, 'delta': new - old})
    rows = []
    for (quantity, scenario), values in sorted(grouped.items()):
        assert len(values) == 5 and {r['fold'] for r in values} == set(range(5))
        deltas = [r['delta'] for r in values]
        rows.append({'quantity': quantity, 'scenario': scenario, 'fold_values': values,
            'mean_across_fixed_artifacts': mean(deltas), 'minimum': min(deltas), 'maximum': max(deltas),
            'positive_artifacts': sum(v > 0 for v in deltas), 'negative_artifacts': sum(v < 0 for v in deltas)})
    assert len(rows) == 27
    out = BASE / 'fitted_scalar_reviews' / args.candidate / 'v1/RESULTS.json'
    atomic_json(out, {'created_utc': utc_now(), 'status': 'PASS', 'candidate': args.candidate,
        'fixed_artifacts': 5, 'rows': rows, 'inputs_sha256': inputs,
        'new_fits': 0, 'new_predictions': 0, 'new_draws': 0,
        'scope': 'Post-hoc descriptive review of existing scalar tables; fixed overlapping fits and common synthetic seeds; no aggregate confidence interval or independent-fold claim; does not override the complete 97-artifact summary or quality gates.',
        'source_sha256': digest(Path(__file__))}, immutable=True)
    progress('Completed fitted artifacts reviewed', args.candidate + ': five original scalar tables per quantity; no resampling')
    print('NEXT5 FIVE FITTED SCALAR REVIEW PASS', args.candidate, len(rows), flush=True)


if __name__ == '__main__':
    main()
