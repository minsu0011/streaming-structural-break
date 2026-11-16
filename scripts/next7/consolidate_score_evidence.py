"""Unify completed score evidence with explicitly different median estimands."""
from pathlib import Path
import csv
import os
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['NUMBA_CACHE_DIR'] = str(ROOT / 'artifacts/next7/numba_cache')
import numpy as np
from src.next7.common import BASE, read_json, save, csv_write, digest, utc_now, activity, note
from src.next7.uncertainty import paired_summary


def rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def main():
    assert not (BASE / 'SCORE_EVIDENCE_CONSOLIDATED.json').exists()
    input_hashes = {}
    scores, generations = {}, {}
    with activity('Consolidate five-fold scores and unambiguous paired intervals', 'FROZEN_SCORE_FORENSIC', 'F5_CAUSAL_NESTED_INTEGRATION'):
        for generation in ('G1', 'G2', 'G3'):
            path = BASE / f'CV_RESULTS_NEXT7_{generation}.csv'
            input_hashes[path.relative_to(ROOT).as_posix()] = digest(path)
            for row in rows(path):
                method = row['method'].replace('AMPLITUDE_0.9_M0', 'AMPLITUDE_0_9_M0')
                fold = int(row['fold'])
                score = float(row['TS_AUC'])
                if method not in scores:
                    scores[method] = np.full(5, np.nan)
                    generations[method] = generation
                if np.isfinite(scores[method][fold]):
                    assert scores[method][fold] == score
                scores[method][fold] = score
        legacy_path = ROOT / 'artifacts/next6/OUTER_SCORES.json'
        legacy = read_json(legacy_path)
        np.testing.assert_array_equal(legacy['M0'], scores['M0'])
        scores['M3'] = np.asarray(legacy['M3'], float)
        generations['M3'] = 'FROZEN_NEXT6_REFERENCE'
        input_hashes[legacy_path.relative_to(ROOT).as_posix()] = digest(legacy_path)
        baseline = scores['M0']
        bootstraps = {}
        for method in ('M0', *read_json(BASE / 'G1_FINALISTS_FROZEN.json')['selected'], 'G2_EVENT', 'G2_FULL', 'G3_PAIRWISE_FULL'):
            generation = 'g3' if method == 'G3_PAIRWISE_FULL' else 'g2' if method.startswith('G2_') else 'g1'
            matrix = np.empty((2001, 5))
            for fold in range(5):
                path = BASE / f'bootstrap/{generation}/{method}/fold_{fold}.npy'
                receipt = read_json(path.with_suffix('.json'))
                assert digest(path) == receipt['result_sha256']
                input_hashes[path.relative_to(ROOT).as_posix()] = digest(path)
                matrix[:, fold] = np.load(path, allow_pickle=False)
            np.testing.assert_allclose(matrix[0], scores[method], rtol=0, atol=1e-12)
            bootstraps[method] = matrix
        comparisons, intervals, fold_rows = [], [], []
        summary_by_method = {}
        for method, score in scores.items():
            assert score.shape == (5,) and np.isfinite(score).all()
            delta = score - baseline
            paired = None
            if method in bootstraps and method != 'M0':
                paired = paired_summary(bootstraps['M0'], bootstraps[method])
                intervals.append({'candidate': method, 'reference': 'M0', **paired,
                    'fixed_fitted_OOF_conditional': True, 'adaptive_selection_not_corrected': True,
                    'new_random_draws': 0, 'whole_series_within_original_fold': True})
                np.testing.assert_allclose([paired['estimate_mean_matched_delta'], paired['estimate_median_matched_fold_delta']],
                    [delta.mean(), np.median(delta)], rtol=0, atol=1e-12)
            record = {'method': method, 'generation_or_reference': generations[method],
                'RP_series': 6400, 'original_grouped_folds': 5, 'mean_TS_AUC': float(score.mean()),
                'median_fold_TS_AUC': float(np.median(score)), 'worst_fold_TS_AUC': float(score.min()),
                'mean_matched_delta_M0': float(delta.mean()), 'median_matched_fold_delta_M0': float(np.median(delta)),
                'difference_of_fold_score_medians_M0': float(np.median(score) - np.median(baseline)),
                'worst_matched_fold_delta_M0': float(delta.min()), 'positive_matched_folds': int(np.sum(delta > 0)),
                'mean_delta_CI_low': paired['mean_delta_ci_2_5'] if paired else None,
                'mean_delta_CI_high': paired['mean_delta_ci_97_5'] if paired else None,
                'probability_mean_delta_positive': paired['probability_mean_delta_positive'] if paired else None,
                'existing_bootstrap_draws': 2000 if paired else 0,
                'new_NEXT7_scientific_candidate': method not in ('M0', 'M3', 'AMPLITUDE_0_9_M0'),
                'standalone_diagnostic': method in ('CURRENT_ONLY', 'EVENT_ONLY', 'SCALE_ONLY'),
                'primary_mean_effect_gate_pass': bool(delta.mean() >= .0012),
                'generation_mean_effect_gate_pass': bool(delta.mean() >= .0025),
                'robustness_mean_noninferiority_gate_pass': bool(delta.mean() >= -.0005),
                'robustness_worst_fold_gate_pass': bool(delta.min() >= 0),
                'full_promotion_decision_deferred_to_final_audit': True}
            for fold in range(5):
                record[f'fold_{fold}_TS_AUC'] = float(score[fold])
                record[f'fold_{fold}_matched_delta_M0'] = float(delta[fold])
                fold_rows.append({'method': method, 'fold': fold, 'TS_AUC': float(score[fold]), 'matched_delta_M0': float(delta[fold])})
            comparisons.append(record)
            summary_by_method[method] = record
        assert sum(r['new_NEXT7_scientific_candidate'] for r in comparisons) == 15
        for candidate, reference in [('G2_EVENT', 'M_EVENT'), ('G2_FULL', 'M_FULL'), ('G3_PAIRWISE_FULL', 'G2_FULL')]:
            intervals.append({'candidate': candidate, 'reference': reference,
                **paired_summary(bootstraps[reference], bootstraps[candidate]), 'fixed_fitted_OOF_conditional': True,
                'adaptive_selection_not_corrected': True, 'new_random_draws': 0, 'whole_series_within_original_fold': True})
        csv_write('MODEL_COMPARISON_NEXT7.csv', comparisons)
        csv_write('MATCHED_FIVE_FOLD_RESULTS_NEXT7.csv', fold_rows)
        csv_write('PAIRED_UNCERTAINTY_NEXT7.csv', intervals)
        save('MEDIAN_ESTIMAND_CLARIFICATION_NEXT7.json', {'status': 'PASS', 'created_utc': utc_now(),
            'legacy_field': 'G1/G2 paired-bootstrap estimate_median_of_folds_delta',
            'actual_legacy_meaning': 'median(candidate fold scores) minus median(reference fold scores)',
            'required_matched_median': 'median(candidate fold score minus matched reference fold score)',
            'legacy_files_preserved': True, 'new_tables_label_both_separately': True,
            'existing_draw_arrays_reused': True, 'draws_added': 0,
            'I4_example': {k: summary_by_method['I4_EVENT_SCALE_TREE'][k] for k in
                ('median_matched_fold_delta_M0', 'difference_of_fold_score_medians_M0')}})
        save('SCORE_EVIDENCE_CONSOLIDATED.json', {'status': 'COMPLETE', 'created_utc': utc_now(),
            'new_candidate_count': 15, 'all_next7_mean_deltas_negative': all(r['mean_matched_delta_M0'] < 0
                for r in comparisons if r['new_NEXT7_scientific_candidate']),
            'learned_recipes': 5, 'scores_not_recomputed_from_labels': True, 'new_draws': 0,
            'source_sha256': digest(__file__), 'inputs_sha256': input_hashes,
            'scope': 'Completed predictive evidence only. Remaining robustness, runtime and final audit are separate requirements.'})
    note('Score evidence consolidated', '15 NEXT7 candidates, same paired draws, median matched differences separated from difference of medians', 'F5_CAUSAL_NESTED_INTEGRATION')


if __name__ == '__main__':
    main()
