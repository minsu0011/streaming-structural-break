"""Recompute every published PRIMARY fracture from the original fixed counts.

This is a numerical reproduction audit, not an additional bootstrap sample or
another opportunity to tune eligibility. It runs only in the final audit phase.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import re
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from numba import set_num_threads
from scripts.verify_scorer import load_official, official_score
from src.next5.artifacts import BASE, atomic_json, read_json, digest, utc_now, progress
from src.next5.data import prepare_research, admitted_npz
from src.next5.bootstrap import PreparedAUC, paired_summary
from src.next5.fractures import masks, pair_mass
from src.next5.references import primary
from src.scoring.ts_auc import ts_auc


def read_table(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def close(value, expected):
    assert abs(float(value) - float(expected)) < 1e-14, (value, expected)


def old_draws(cid, fold, scope, *, baseline=False):
    if scope == 'all_time':
        root = BASE / 'bootstrap' / ('ResearchPool_BL80' if baseline else cid)
        name = f'fold_{fold}.npy'
        sha = (read_json(root / f'fold_{fold}.json')['sha256'] if baseline
               else read_json(root / 'RESULTS.json')['files_sha256'][name])
    else:
        root = BASE / 'forensic' / ('ResearchPool_BL80' if baseline else cid)
        name = f'fold_{fold}_{scope}.npy'
        sha = (read_json(root / (name + '.json'))['sha256'] if baseline
               else read_json(root / 'RESULTS.json')['files_sha256'][name])
    assert digest(root / name) == sha
    return np.load(root / name, allow_pickle=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    if not re.fullmatch('[A-Za-z0-9_]+', args.tag):
        raise ValueError('Safe immutable audit tag required')
    session = read_json(BASE / 'SESSION.json')
    assert session.get('audit_started_utc')
    out = BASE / 'final_audit' / args.tag / 'frozen_statistics'
    out.mkdir(parents=True, exist_ok=False)
    policy_path = BASE / 'diagnostics/FRACTURE_POLICY.json'
    policy = read_json(policy_path)
    assert digest(BASE / 'bootstrap/COUNTS_LOCK.json') == policy['counts_lock_sha256']
    assert digest(BASE / 'diagnostics/RESEARCH_DIAGNOSTIC_LOCK.json') == policy['diagnostic_lock_sha256']
    for name, sha in policy['source_sha256'].items():
        assert digest(ROOT / name) == sha
    atomic_json(out / 'START.json', {'created_utc': utc_now(), 'source_sha256': digest(Path(__file__)),
        'audit_started_utc': session['audit_started_utc'], 'new_fits': 0, 'new_draws': 0,
        'scope': 'All 19 full candidates, five PRIMARY folds, all 26 frozen scopes; original 2000 count rows'}, immutable=True)
    set_num_threads(2)
    data = prepare_research()
    metadata = data.rows()
    baseline, _, reference_proof = primary('full', 'PRIMARY', metadata, data.guard)
    diagnostic = BASE / 'diagnostics'
    lock = read_json(diagnostic / 'RESEARCH_DIAGNOSTIC_LOCK.json')
    sigpath = diagnostic / 'RESEARCH_SIGNATURES.npz'
    assert digest(sigpath) == lock['files_sha256'][sigpath.name]
    signatures = admitted_npz(sigpath, data.guard, 'Final audit RP-only evaluation signatures')
    order = np.argsort(signatures['dataset_id'])
    signatures = {k: v[order] for k, v in signatures.items()}
    count_lock = read_json(BASE / 'bootstrap/COUNTS_LOCK.json')
    count_arrays = {}
    for receipt in count_lock['folds']:
        path = ROOT / receipt['path']
        assert digest(path) == receipt['sha256']
        values = admitted_npz(path, data.guard, 'Final audit original paired count rows')
        assert values['counts'].shape == (2001, 1280)
        assert np.all(values['counts'][0] == 1) and np.all(values['counts'].sum(axis=1) == 1280)
        count_arrays[receipt['fold']] = values
    official = load_official()
    original_official = read_json(BASE / 'preflight/PREEXISTING_SOURCE_AND_REPORT_SNAPSHOT.json')['files']
    # The vendored scorer has its own original provenance; this audit executes it
    # without modifying scoring logic and substitutes only in-memory I/O.
    official_sha = digest(ROOT / 'vendor/official/scoring.py')
    scorer_proof = read_json(ROOT / 'artifacts/validation/SCORER_PARITY.json')
    assert official_sha == scorer_proof['official_source_sha256']
    assert digest(ROOT / 'scripts/verify_scorer.py') == original_official['scripts/verify_scorer.py']['sha256']
    candidates = sorted(p.parent.parent.name for p in (BASE / 'experiments').glob('*/full/RESULTS.json'))
    assert len(candidates) == 19
    fresh_baseline = {}
    baseline_identity = {}
    checked_rows = []
    official_rows = []
    candidate_reports = []
    max_score_error = 0.
    for cid in candidates:
        source = BASE / 'experiments' / cid / 'full'
        old_root = BASE / 'forensic' / cid
        result = read_json(source / 'RESULTS.json')
        forensic = read_json(old_root / 'RESULTS.json')
        for name in ('PAIRED_FRACTURES.csv', 'FOLD_SCORES.csv', 'SUPPORT.csv'):
            assert digest(old_root / name) == forensic['files_sha256'][name]
        old_summary = {r['scope']: r for r in read_table(old_root / 'PAIRED_FRACTURES.csv')}
        old_folds = {(int(r['fold']), r['scope']): r for r in read_table(old_root / 'FOLD_SCORES.csv')}
        old_support = {(int(r['fold']), r['scope']): r for r in read_table(old_root / 'SUPPORT.csv')}
        new_by_scope, old_by_scope = {}, {}
        for fold in range(5):
            directory = source / f'fold_{fold}'
            receipt = read_json(directory / 'RESULTS.json')
            path = directory / 'predictions.npz'
            assert digest(path) == receipt['predictions_file_sha256']
            values = admitted_npz(path, data.guard, 'Final audit full RP held-out prediction coordinates')
            rows = values['global_row']
            np.testing.assert_array_equal(rows, np.flatnonzero(metadata['fold'] == fold))
            ids = metadata['dataset_id'][rows]
            np.testing.assert_array_equal(values['dataset_id'], ids)
            np.testing.assert_array_equal(values['baseline'], baseline[rows])
            index = np.searchsorted(signatures['dataset_id'], ids)
            np.testing.assert_array_equal(signatures['dataset_id'][index], ids)
            y = metadata['target'][rows]
            times = metadata['time_online'][rows]
            tau = signatures['tau'][index]
            np.testing.assert_array_equal(y, ((tau >= 0) & (times >= tau)).astype(np.uint8))
            scopes = masks(y, times, tau, signatures['historical_regime'][index],
                           signatures['signature'][index], signatures['observed_scale_direction'][index])
            assert set(scopes) == set(old_summary) and len(scopes) == 26
            draws = count_arrays[fold]
            counts = draws['counts']
            code = np.searchsorted(draws['dataset_id'], ids).astype(np.int32)
            np.testing.assert_array_equal(draws['dataset_id'][code], ids)
            full_mass = pair_mass(y, times)
            contributions = {'time_': 0., 'age_': 0.}
            masses = {'time_': 0, 'age_': 0}
            for name, mask in scopes.items():
                key = (fold, name)
                before = old_folds[key]
                support = old_support[key]
                points = int(mask.sum())
                mass = pair_mass(y[mask], times[mask])
                positive = len(np.unique(ids[mask & (y == 1)]))
                nulls = len(np.unique(ids[mask & (tau < 0)]))
                break_present = len(np.unique(ids[mask & (tau >= 0)]))
                for field, value in [('points', points), ('pairs', mass), ('positive_series', positive),
                                     ('null_series', nulls), ('break_series_present', break_present)]:
                    assert int(support[field]) == value
                p = values['prediction'][mask]
                b = values['baseline'][mask]
                score = float(ts_auc(y[mask], p, times[mask]))
                refscore = float(ts_auc(y[mask], b, times[mask]))
                close(score, before['TS_AUC'])
                close(refscore, before['BL80_TS_AUC'])
                close(score - refscore, before['delta'])
                contribution = mass / full_mass * (score - refscore)
                close(contribution, before['weighted_contribution_delta'])
                assert int(before['pairs']) == mass and int(before['all_time_pairs']) == full_mass
                for prefix in contributions:
                    if name.startswith(prefix):
                        contributions[prefix] += contribution
                        masses[prefix] += mass
                identity = hashlib.sha256(rows[mask].tobytes() + b.tobytes()).hexdigest()
                if key not in fresh_baseline:
                    fresh = PreparedAUC.create(y[mask], b, times[mask], code[mask], counts.shape[1]).evaluate(counts) if points else np.full(2001, .5)
                    np.testing.assert_array_equal(fresh, old_draws(cid, fold, name, baseline=True))
                    fresh_baseline[key] = fresh
                    baseline_identity[key] = identity
                else:
                    assert baseline_identity[key] == identity
                new = PreparedAUC.create(y[mask], p, times[mask], code[mask], counts.shape[1]).evaluate(counts) if points else np.full(2001, .5)
                np.testing.assert_array_equal(new, old_draws(cid, fold, name))
                old = fresh_baseline[key]
                max_score_error = max(max_score_error, abs(float(new[0]) - score), abs(float(old[0]) - refscore))
                close(new[0], score)
                close(old[0], refscore)
                new_by_scope.setdefault(name, []).append(new)
                old_by_scope.setdefault(name, []).append(old)
                checked_rows.append({'candidate': cid, 'fold': fold, 'scope': name, 'points': points,
                    'pairs': mass, 'positive_series': positive, 'null_series': nulls,
                    'TS_AUC': score, 'BL80_TS_AUC': refscore, 'weighted_contribution_delta': contribution,
                    'original_counts_sha256': count_lock['folds'][fold]['sha256'],
                    'recomputed_draw_values_sha256': hashlib.sha256(new.tobytes()).hexdigest(),
                    'all_2001_values_exactly_equal': True})
                if name == 'all_time':
                    idx = pd.MultiIndex.from_arrays([ids, times], names=['id', 'time'])
                    frame = pd.DataFrame({'prediction': values['prediction']}, index=idx).sort_index()
                    targets = pd.DataFrame({'target': y}, index=idx).sort_index()
                    official_value = official_score(official, frame, targets)
                    close(official_value, score)
                    official_rows.append({'candidate': cid, 'fold': fold, 'rows': len(rows),
                        'official_TS_AUC': official_value, 'local_TS_AUC': score, 'absolute_error': abs(official_value - score)})
                progress('Final exact fixed-count statistical audit', cid + ' fold ' + str(fold) + ' ' + name,
                         phase='FINAL_AUDIT', counts_as_research=False)
            for prefix in contributions:
                assert masses[prefix] == full_mass
                assert abs(contributions[prefix] - receipt['delta']) < 1e-12
            print('NEXT5 FINAL STATISTICS', cid, 'fold', fold, 'all 26 scopes exact', flush=True)
        for name in scopes:
            new = np.asarray(new_by_scope[name]).T
            old = np.asarray(old_by_scope[name]).T
            summary = paired_summary(old, new)
            before = old_summary[name]
            for field, value in summary.items():
                close(value, before[field])
            close(new[0].mean(), before['candidate_mean'])
            close(old[0].mean(), before['baseline_mean'])
            support = [r for r in checked_rows if r['candidate'] == cid and r['scope'] == name]
            positives = sum(r['positive_series'] for r in support)
            nulls = sum(r['null_series'] for r in support)
            supported = positives >= policy['supported_positive_series_min'] and (
                not name.startswith('regime_') or nulls >= policy['supported_H_null_series_min'])
            assert (before['supported'] == 'True') == supported
            assert int(before['positive_series']) == positives and int(before['null_series']) == nulls
            assert int(before['points']) == sum(r['points'] for r in support)
            close(np.mean([r['weighted_contribution_delta'] for r in support]), before['mean_weighted_contribution_delta'])
            if name == 'all_time':
                bootstrap = read_json(BASE / 'bootstrap' / cid / 'RESULTS.json')
                assert summary == bootstrap['paired_all_time']
                for field, value in [('mean', new[0].mean()), ('median', np.median(new[0])), ('worst', new[0].min()),
                                     ('mean_delta', (new[0] - old[0]).mean()),
                                     ('median_delta', np.median(new[0]) - np.median(old[0])),
                                     ('worst_delta', new[0].min() - old[0].min())]:
                    close(value, result[field])
                gate = read_json(ROOT / 'configs/next5/RESEARCH_POLICY_LOCK.json')['performance_gate']
                numeric = bool(result['mean_delta'] >= gate['mean_delta_min'] and
                    result['median_delta'] >= gate['median_delta_min'] and result['worst_delta'] >= gate['worst_delta_min'])
                admitted = bool(numeric and summary['ci_2_5'] > gate['ci_lower_strict_min'] and
                    summary['probability_delta_positive'] >= gate['bootstrap_probability_min'])
                assert numeric == bootstrap['performance_numeric_admission']
                assert admitted == bootstrap['performance_primary_and_bootstrap_admission']
        candidate_reports.append({'candidate': cid, 'folds': 5, 'scopes': 26,
            'replicates': 2000, 'all_saved_draws_bitwise_equal': True, 'official_full_fold_scores_matched': True,
            'time_age_pair_mass_conservation': True, 'support_counts_and_flags_matched': True,
            'full_result_sha256': digest(source / 'RESULTS.json'), 'forensic_result_sha256': digest(old_root / 'RESULTS.json')})
        atomic_json(out / (cid + '.json'), {'created_utc': utc_now(), 'status': 'PASS', **candidate_reports[-1]}, immutable=True)
    assert len(checked_rows) == 19 * 5 * 26 and len(official_rows) == 19 * 5
    for name, rows in [('RECOMPUTED_STATISTICS.csv', checked_rows), ('OFFICIAL_SCORE_PARITY.csv', official_rows)]:
        with (out / name).open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    data.guard.check_state()
    atomic_json(out / 'RESULTS.json', {
        'created_utc': utc_now(), 'status': 'PASS', 'audit_started_utc': session['audit_started_utc'],
        'candidates': candidate_reports, 'candidate_fold_scopes': len(checked_rows),
        'baseline_fold_scopes_recomputed_once': len(fresh_baseline), 'official_full_fold_comparisons': len(official_rows),
        'maximum_rank_kernel_vs_local_score_error': max_score_error,
        'maximum_official_vs_local_score_error': max(r['absolute_error'] for r in official_rows),
        'all_original_draw_arrays_bitwise_equal': True, 'all_reported_summaries_matched': True,
        'new_or_additional_random_draws': 0, 'new_fits': 0, 'real_data_scope': 'ResearchPool only',
        'Shadow_rows': 0, 'LOCAL_SEAL_rows': 0, 'eligibility_rules_changed': False,
        'official_scorer_sha256': official_sha, 'reference_proof': reference_proof,
        'policy_sha256': digest(policy_path), 'counts_lock_sha256': digest(BASE / 'bootstrap/COUNTS_LOCK.json'),
        'source_sha256': digest(Path(__file__)),
        'files_sha256': {p.name: digest(p) for p in out.iterdir() if p.is_file()},
    }, immutable=True)
    progress('All published PRIMARY statistics reproduced', '2470 candidate fold/scopes, 130 baseline fold/scopes, 95 actual official-score comparisons',
             phase='FINAL_AUDIT', counts_as_research=False)


if __name__ == '__main__':
    main()
