"""Release G2 scores once, after all five outer predictions are frozen."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.next7.common import BASE, save, read_json, digest, csv_write, utc_now, activity, require_guard, note
from src.next6.data import source
from src.scoring.ts_auc import ts_auc


def main():
    guard = require_guard()
    frozen = read_json(BASE / 'G2_ALL_OUTER_PREDICTIONS_FROZEN.json')
    assert frozen['status'] == 'FROZEN' and frozen['all_five_completed_before_G2_scores']
    assert not (BASE / 'G2_OUTER_SCORE_RELEASE.json').exists(), 'G2 score release is immutable'
    save('G2_BOOTSTRAP_SELECTION_POLICY.json', {'status': 'FROZEN', 'created_utc': utc_now(),
        'before_G2_outer_score_release': True, 'candidates': ['G2_EVENT', 'G2_FULL'],
        'draw_count': '5000 if either mean matched delta >=.0025; otherwise2000',
        'references': ['M0', 'G1_M_EVENT', 'G1_M_FULL'],
        'resampling': 'whole series within original grouped fold; identical counts across comparisons',
        'no_post_interval_extension': True, 'adaptive_selection_not_corrected': True,
        'budget_scope': 'Two already reserved full candidates, one already reserved learned recipe; no new candidate'})
    for record in frozen['folds']:
        for name, sha in record['prediction_sha256'].items():
            assert digest(BASE / f'g2/fold_{record["fold"]}' / name) == sha
    save('G2_OUTER_SCORE_RELEASE.json', {'created_utc': utc_now(), 'status': 'FROZEN_PREDICTIONS_RELEASED',
        'prediction_freeze_sha256': digest(BASE / 'G2_ALL_OUTER_PREDICTIONS_FROZEN.json'),
        'bootstrap_policy_sha256': digest(BASE / 'G2_BOOTSTRAP_SELECTION_POLICY.json'),
        'G2_admission_cut_and_recipe_now_closed_to_tuning': True})
    with activity('G2 single outer score release', 'SCIENTIFIC_EVALUATION', 'F5_CAUSAL_NESTED_INTEGRATION'):
        data = source()
        metadata = data.rows()
        rows_out, scores = [], {}
        for fold in range(5):
            directory = BASE / f'g2/fold_{fold}'
            for path in (directory / 'fixed_predictions.npz', directory / 'G2_FULL/predictions.npz'):
                with np.load(path, allow_pickle=False) as values:
                    rows = values['global_row']
                    guard.admit(np.unique(values['dataset_id']), purpose='G2 single score release')
                    np.testing.assert_array_equal(metadata['dataset_id'][rows], values['dataset_id'])
                    assert {guard.split.assignment(int(s))[1] for s in np.unique(values['dataset_id'])} == {fold}
                    for key in values.files:
                        if key in ('global_row', 'dataset_id'):
                            continue
                        method = 'G2_FULL' if key == 'prediction' else key
                        score = float(ts_auc(metadata['target'][rows], values[key], metadata['time_online'][rows]))
                        scores.setdefault(method, {})[str(fold)] = score
                        rows_out.append({'method': method, 'fold': fold, 'TS_AUC': score,
                            'series': len(np.unique(values['dataset_id'])), 'rows': len(rows),
                            'strict_nested_correction': method == 'G2_FULL', 'no_outer_label_fit': True})
            print(utc_now(), 'G2 outer scored', fold, flush=True)
        g1 = read_json(BASE / 'G1_OUTER_SCORES.json')['scores']
        baseline = np.array([scores['M0'][str(f)] for f in range(5)])
        np.testing.assert_array_equal(baseline, [g1['M0'][str(f)] for f in range(5)])
        np.testing.assert_array_equal(baseline, [scores['AMPLITUDE_0_9_M0'][str(f)] for f in range(5)])
        summary = []
        for method, values in scores.items():
            s = np.array([values[str(f)] for f in range(5)])
            delta = s - baseline
            summary.append({'method': method, 'folds': 5, 'mean_TS_AUC': float(s.mean()),
                'median_TS_AUC': float(np.median(s)), 'worst_TS_AUC': float(s.min()),
                'mean_delta_M0': float(delta.mean()), 'median_delta_M0': float(np.median(delta)),
                'worst_fold_delta_M0': float(delta.min()), 'positive_matched_folds': int((delta > 0).sum()),
                'conditional_DEVELOPMENT_only': True, 'promotion_not_decided_by_mean_alone': True})
        for r in rows_out:
            r['delta_M0'] = r['TS_AUC'] - float(baseline[r['fold']])
        csv_write('CV_RESULTS_NEXT7_G2.csv', rows_out)
        csv_write('MODEL_COMPARISON_NEXT7_G2.csv', summary)
        save('G2_OUTER_SCORES.json', {'created_utc': utc_now(), 'scores': scores, 'summaries': summary,
            'M0_frozen_parity': 'PASS', 'new_full_scientific_candidates': 2})
        strong = any(r['mean_delta_M0'] >= .0025 for r in summary if r['method'].startswith('G2_'))
        save('G2_BOOTSTRAP_CANDIDATES_FROZEN.json', {'status': 'FROZEN', 'created_utc': utc_now(),
            'selected': ['G2_EVENT', 'G2_FULL'], 'draws': 5000 if strong else 2000,
            'draw_count_decided_before_bootstrap': True, 'not_a_promotion': True})
        usage = read_json(BASE / 'runtime/CANDIDATE_USAGE.json')
        usage.update(full_completed=14, updated_utc=utc_now(), G2_scored=['G2_EVENT', 'G2_FULL'])
        save('runtime/CANDIDATE_USAGE.json', usage, False)
        print(summary, flush=True)
    guard.check_state()
    note('G2 scores frozen', 'Quantify paired uncertainty and H support; preserve all negative results', 'F4_NULL_GROWTH_CORRECTION')


if __name__ == '__main__':
    main()
