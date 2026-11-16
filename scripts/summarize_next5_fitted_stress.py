"""Require every frozen fitted artifact before combining stress diagnostics."""
from pathlib import Path
import argparse
import csv
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.next5.artifacts import BASE, atomic_json, read_json, digest, utc_now, progress
from src.next5.fitted_fold_summary import fixed_artifact_mean, describe_fold_values
from src.next5.scheduler import register_job

TABLES = (
    'CALIBRATION.csv', 'COUNTEREXAMPLE_SCORES.csv', 'PAIRED_COUNTEREXAMPLES.csv',
    'COUNTEREXAMPLE_ALARMS.csv', 'LONG_NULL_RESULTS.csv', 'PAIRED_ALARMS.csv',
    'OUTLIER_LAG_RESPONSE.csv', 'OUTLIER_RESPONSE_SUMMARY.csv',
    'PAIRED_OUTLIER_RESPONSE.csv', 'FEATURE_IMPULSE_RESPONSE.csv',
)


def read_table(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_table(path, rows):
    if not rows:
        raise ValueError('An expected fitted stress table is empty')
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def register():
    out = BASE / 'synthetic/fitted_summary'
    out.mkdir(parents=True, exist_ok=True)
    coverage = read_json(BASE / 'synthetic/FITTED_COVERAGE_LOCK.json')
    sources = ('scripts/summarize_next5_fitted_stress.py', 'src/next5/fitted_fold_summary.py',
               'src/next5/bootstrap.py', 'tests/test_next5_fitted_fold_summary.py')
    atomic_json(out / 'POLICY_LOCK.json', {
        'created_utc': utc_now(), 'fitted_coverage_sha256': digest(BASE / 'synthetic/FITTED_COVERAGE_LOCK.json'),
        'sources_sha256': {name: digest(ROOT / name) for name in sources},
        'coverage': 'Every one of the 97 preregistered fitted artifacts; failed-screen M2 remains screen-only',
        'new_fits': 0, 'new_random_draws': 0, 'real_point_predictions_read': False,
        'point_tables': 'Retain all fitted-fold rows, with matched references and calibration scope',
        'TS_AUC_summary': 'Equal mean of fixed artifact scores with the original common generating-seed bootstrap rows; not an ensemble prediction score',
        'alarm_and_impulse_summary': 'Descriptive mean, minimum, maximum and signs of point differences; no confidence interval formed from overlapping folds',
        'deployment_scope': 'CV artifacts only. No actual new all-ResearchPool or all-DEV model exists or is implied.',
        'primary_gate_relaxation': False,
    }, immutable=True)
    register_job({'id': 'SYN_SUMMARY_CONTRACTS', 'kind': 'validation', 'priority': 35,
        'argv': ['-m', 'pytest', 'tests/test_next5_fitted_fold_summary.py', '-q',
                 '--junitxml=artifacts/next5/synthetic/fitted_summary/CONTRACTS.xml'],
        'description': 'Check that common seed uncertainty is preserved and overlapping folds are not independent resamples',
        'retry_safe': False})
    dependencies = ['SYN_SUMMARY_CONTRACTS'] + [
        f"SYN_ANALYZE_{r['candidate']}_{r['scope']}_{r['fold']}" for r in coverage['fitted_artifacts']]
    register_job({'id': 'SYN_ALL_FITTED_SUMMARY', 'kind': 'validation', 'priority': 91,
        'depends_on_success': dependencies, 'argv': ['scripts/summarize_next5_fitted_stress.py'],
        'description': 'Complete fixed-fold stress coverage and conditional common-seed score summaries; no new models or seed draws',
        'retry_safe': False})
    progress('All-fitted-artifact stress summary protocol frozen', '97 artifact coverage, paired seed rows preserved, no independent-fold error bars')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--register', action='store_true')
    args = parser.parse_args()
    if args.register:
        register()
        return
    parent = BASE / 'synthetic/fitted_summary'
    policy = read_json(parent / 'POLICY_LOCK.json')
    for name, sha in policy['sources_sha256'].items():
        assert digest(ROOT / name) == sha
    coverage_path = BASE / 'synthetic/FITTED_COVERAGE_LOCK.json'
    assert digest(coverage_path) == policy['fitted_coverage_sha256']
    coverage = read_json(coverage_path)['fitted_artifacts']
    expected = {}
    tables = {name: [] for name in TABLES}
    receipts = []
    for record in coverage:
        cid, scope, fold = record['candidate'], record['scope'], record['fold']
        root = BASE / 'synthetic/candidates' / cid / scope / ('fold_' + str(fold)) / 'ANALYSIS'
        result = read_json(root / 'RESULTS.json')
        assert result['status'] == 'PASS' and result['cohorts'] == 34
        assert (result['candidate'], result['scope'], result['fold']) == (cid, scope, fold)
        assert result['replicates'] == 2000
        expected.setdefault((cid, scope), []).append(fold)
        for name in TABLES:
            assert digest(root / name) == result['files_sha256'][name]
            rows = read_table(root / name)
            assert all((r['candidate'], r['scope'], int(r['fold'])) == (cid, scope, fold) for r in rows)
            tables[name].extend(rows)
        receipts.append({'candidate': cid, 'scope': scope, 'fold': fold,
                         'analysis_sha256': digest(root / 'RESULTS.json')})
    assert len(receipts) == 97 and len(expected) == 20
    for (cid, scope), folds in expected.items():
        assert sorted(folds) == (list(range(5)) if scope == 'full' else [0, 4])
    out = parent / 'complete_v1'
    out.mkdir(exist_ok=False)
    files = {}
    for name, rows in tables.items():
        write_table(out / name, rows)
        files[name] = digest(out / name)
    score_groups = {}
    for row in tables['PAIRED_COUNTEREXAMPLES.csv']:
        key = (row['candidate'], row['scope'], row['scenario'], row['reference'], int(row['horizon']))
        score_groups.setdefault(key, []).append(row)
    score_summaries = []
    for (cid, scope, scenario, role, horizon), rows in sorted(score_groups.items()):
        old, new = {}, {}
        for row in rows:
            fold = int(row['fold'])
            assert fold not in old
            root = BASE / 'synthetic/candidates' / cid / scope / ('fold_' + str(fold)) / 'ANALYSIS'
            name = scenario + '_CANDIDATE_TS_AUC_DRAWS.npy'
            receipt = read_json(root / 'RESULTS.json')
            assert digest(root / name) == receipt['files_sha256'][name]
            new[fold] = np.load(root / name, allow_pickle=False)
            ref = BASE / 'synthetic/reference_bootstrap' / scope / ('fold_' + str(fold)) / (scenario + '_' + role + '_' + str(horizon))
            ref_receipt = read_json(ref.with_suffix('.json'))
            assert digest(ref.with_suffix('.npy')) == ref_receipt['sha256']
            assert ref_receipt['identity']['execution_sha256'] == receipt['execution_sha256']
            assert ref_receipt['identity']['counts_sha256'] == receipt['seed_counts_sha256']
            old[fold] = np.load(ref.with_suffix('.npy'), allow_pickle=False)
            assert abs(new[fold][0] - float(row['candidate_TS_AUC'])) < 1e-14
            assert abs(old[fold][0] - float(row['reference_TS_AUC'])) < 1e-14
        score_summaries.append({'candidate': cid, 'scope': scope, 'scenario': scenario, 'reference': role,
            'horizon': horizon, **fixed_artifact_mean(old, new, expected[(cid, scope)])})
        progress('Fixed-artifact stress summary', cid + ' ' + scenario + ' vs ' + role + '; existing seed draws only')
    write_table(out / 'FIXED_ARTIFACT_MEAN_TS_AUC.csv', score_summaries)
    for name, keys, target in (
        ('PAIRED_ALARMS.csv', ['candidate', 'scope', 'scenario', 'reference', 'horizon', 'alpha', 'event'], 'FOLD_ALARM_HETEROGENEITY.csv'),
        ('PAIRED_OUTLIER_RESPONSE.csv', ['candidate', 'scope', 'scenario', 'reference', 'metric', 'lag_from_first_impulse'], 'FOLD_IMPULSE_HETEROGENEITY.csv'),
    ):
        write_table(out / target, describe_fold_values(tables[name], keys, 'estimate_mean_delta', expected))
        files[target] = digest(out / target)
    files['FIXED_ARTIFACT_MEAN_TS_AUC.csv'] = digest(out / 'FIXED_ARTIFACT_MEAN_TS_AUC.csv')
    atomic_json(out / 'RESULTS.json', {
        'created_utc': utc_now(), 'status': 'PASS', 'fitted_artifacts': 97, 'candidates': 20,
        'source_receipts': receipts, 'policy_sha256': digest(parent / 'POLICY_LOCK.json'),
        'files_sha256': files, 'new_fits': 0, 'new_random_draws': 0,
        'rows_per_table': {name: len(rows) for name, rows in tables.items()},
        'actual_deployment_artifact_claimed': False, 'independent_fold_uncertainty_claimed': False,
        'uncertainty': 'Conditional common generating-seed draws across fixed overlapping fitted artifacts; no calibration or learner refitting uncertainty',
    }, immutable=True)
    progress('All 97 fitted stress artifacts summarized', 'Complete fixed coverage; original paired seed draws and explicit fold heterogeneity')


if __name__ == '__main__':
    main()
