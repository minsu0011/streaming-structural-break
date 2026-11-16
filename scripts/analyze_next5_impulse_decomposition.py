"""Separate affine amplitude dilution from new-expert impulse persistence."""
from pathlib import Path
import argparse
import csv
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next5.artifacts import BASE, atomic_json, read_json, digest, utc_now, progress
from src.next5.scheduler import register_job

LINEAR_METRICS = ('mean_excess_at_horizon1024', 'last64_excess_at_horizon1024',
                  'last64_excess_at_horizon2048', 'post_impulse_mean_excess_at_horizon2048')


def read_table(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_table(path, rows):
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def decomposition(base, expert, combined, weight):
    a, b = weight
    if min(a, b) < 0 or abs(a + b - 1.) > 1e-15:
        raise ValueError('A fixed convex BL80/expert combination is required')
    expected = a * base + b * expert
    if abs(combined - expected) > 1e-14:
        raise ValueError('Linear impulse summary does not match its fixed composite')
    return {'BL80_excess': base, 'expert_excess': expert, 'candidate_excess': combined,
        'BL80_weight': a, 'expert_weight': b, 'affine_BL80_with_constant_expert_excess': a * base,
        'amplitude_only_delta_vs_BL80': (a - 1.) * base,
        'new_expert_excess_above_affine_control': b * expert,
        'total_candidate_delta_vs_BL80': combined - base,
        'linear_reconstruction_absolute_error': abs(combined - expected)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--register', action='store_true')
    args = parser.parse_args()
    parent = BASE / 'impulse_decomposition'
    if args.register:
        atomic_json(parent / 'POLICY_LOCK.json', {
            'created_utc': utc_now(), 'source_sha256': digest(Path(__file__)),
            'purpose': 'Distinguish response-amplitude dilution from genuine new-expert recovery using exact linear identities',
            'timing': 'Post-hoc interpretation control prompted by similar long-lag excess reductions in initial P/R folded stress reports',
            'scope': 'All existing fitted candidates whose frozen combination is a*BL80+b*expert; S4A uses another composition and is explicitly excluded',
            'control': 'a*BL80+b*constant has exactly a times BL80 impulse excess, for any input-invariant constant',
            'metric_boundary': 'A strictly increasing affine score transform preserves exact-real-arithmetic ranking and, with transformed cutoff, alarm events; it need not improve detection',
            'float64_boundary': 'No numerical prediction or bitwise-AUC parity claim is made for a newly evaluated affine transform; no such control predictions are generated',
            'linear_metrics': LINEAR_METRICS, 'recovery_times_decomposed': False,
            'new_fits': 0, 'new_predictions': 0, 'new_random_draws': 0, 'promotion_authority': False,
            'coverage_lock_sha256': digest(BASE / 'synthetic/FITTED_COVERAGE_LOCK.json'),
        }, immutable=True)
        register_job({'id': 'IMPULSE_AFFINE_DECOMPOSITION', 'kind': 'validation', 'priority': 92,
            'depends_on_success': ['SYN_ALL_FITTED_SUMMARY'],
            'argv': ['scripts/analyze_next5_impulse_decomposition.py'],
            'description': 'All eligible fitted-artifact linear impulse summaries: affine amplitude dilution plus added expert excess',
            'retry_safe': False})
        progress('Impulse interpretation control frozen', 'Exact fixed-weight decomposition; no new candidate, predictions, seed, threshold or gate')
        return
    policy = read_json(parent / 'POLICY_LOCK.json')
    assert digest(Path(__file__)) == policy['source_sha256']
    assert digest(BASE / 'synthetic/FITTED_COVERAGE_LOCK.json') == policy['coverage_lock_sha256']
    source = BASE / 'synthetic/fitted_summary/complete_v1'
    receipt = read_json(source / 'RESULTS.json')
    assert receipt['status'] == 'PASS' and receipt['fitted_artifacts'] == 97
    for name in ('OUTLIER_RESPONSE_SUMMARY.csv', 'OUTLIER_LAG_RESPONSE.csv'):
        assert digest(source / name) == receipt['files_sha256'][name]
    summary = read_table(source / 'OUTLIER_RESPONSE_SUMMARY.csv')
    lags = read_table(source / 'OUTLIER_LAG_RESPONSE.csv')
    coverage = read_json(BASE / 'synthetic/FITTED_COVERAGE_LOCK.json')['fitted_artifacts']
    weights, excluded = {}, []
    for record in coverage:
        cid = record['candidate']
        combination = read_json(BASE / 'registrations' / (cid + '.json'))['combination']
        key = (cid, record['scope'], str(record['fold']))
        if set(combination) != {'ResearchPool_BL80', 'new_model'}:
            excluded.append({**record, 'reason': 'Composition is not a BL80/expert convex pair'})
            continue
        weights[key] = (combination['ResearchPool_BL80'], combination['new_model'])
    assert len(weights) == 92 and len(excluded) == 5
    output = []
    for table, metric_field, value_field in ((summary, 'metric', 'estimate'),
                                            (lags, 'lag_from_first_impulse', 'mean_score_excess')):
        groups = {}
        for row in table:
            artifact = (row['candidate'], row['scope'], row['fold'])
            if artifact not in weights or row['model'] not in ('candidate', 'BL80', 'expert'):
                continue
            if metric_field == 'metric' and row['metric'] not in LINEAR_METRICS:
                continue
            key = artifact + (row['scenario'], row[metric_field])
            group = groups.setdefault(key, {})
            assert row['model'] not in group
            group[row['model']] = float(row[value_field])
        for key, values in sorted(groups.items()):
            assert set(values) == {'candidate', 'BL80', 'expert'}
            cid, scope, fold, scenario, metric = key
            output.append({'candidate': cid, 'scope': scope, 'fold': fold, 'scenario': scenario,
                'metric_kind': 'linear_summary' if metric_field == 'metric' else 'mean_at_impulse_lag',
                'metric_or_lag': metric,
                **decomposition(values['BL80'], values['expert'], values['candidate'], weights[(cid, scope, fold)])})
    assert len(output) == 92 * 2 * (4 + 11)
    out = parent / 'complete_v1'
    out.mkdir(parents=True, exist_ok=False)
    write_table(out / 'IMPULSE_EXCESS_DECOMPOSITION.csv', output)
    write_table(out / 'EXCLUDED_COMPOSITIONS.csv', excluded)
    atomic_json(out / 'RESULTS.json', {'created_utc': utc_now(), 'status': 'PASS', 'fitted_artifacts': 92,
        'excluded_fitted_artifacts': 5, 'linear_comparisons': len(output),
        'maximum_linear_reconstruction_error': max(r['linear_reconstruction_absolute_error'] for r in output),
        'new_fits': 0, 'new_predictions': 0, 'new_draws': 0, 'promotion_authority': False,
        'interpretation': 'A lower probability excess alone can reflect affine amplitude dilution. Judge the added expert response with matched calibrated alarms, late genuine breaks and held-out TS-AUC.',
        'policy_sha256': digest(parent / 'POLICY_LOCK.json'), 'complete_stress_summary_sha256': digest(source / 'RESULTS.json'),
        'files_sha256': {p.name: digest(p) for p in out.iterdir() if p.is_file()},
    }, immutable=True)
    progress('Impulse excess fully decomposed', '92 eligible fixed artifacts, exact linear dilution and expert response; nonlinear recovery left intact')


if __name__ == '__main__':
    main()
