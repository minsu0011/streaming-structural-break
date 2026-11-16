"""One score release for the final learned G3 candidate."""
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
    frozen = read_json(BASE / 'G3_ALL_OUTER_PREDICTIONS_FROZEN.json')
    assert frozen['status'] == 'FROZEN' and frozen['all_five_completed_before_G3_scores']
    assert not (BASE / 'G3_OUTER_SCORE_RELEASE.json').exists(), 'G3 scores are immutable'
    for record in frozen['folds']:
        assert digest(BASE / f'g3/fold_{record["fold"]}/predictions.npz') == record['prediction_sha256']
    save('G3_OUTER_SCORE_RELEASE.json', {'created_utc': utc_now(), 'status': 'FROZEN_PREDICTIONS_RELEASED',
        'prediction_freeze_sha256': digest(BASE / 'G3_ALL_OUTER_PREDICTIONS_FROZEN.json'),
        'protocol_and_bootstrap_policy_sha256': digest(BASE / 'G3_PROTOCOL_LOCK.json'),
        'no_G3_loss_or_penalty_or_pair_sample_tuning_after_release': True})
    with activity('G3 single outer score release', 'SCIENTIFIC_EVALUATION', 'F6_TIME_MATCHED_RANK_CORRECTION'):
        data = source()
        metadata = data.rows()
        scores = {'M0': {}, 'G3_PAIRWISE_FULL': {}}
        records = []
        for fold in range(5):
            with np.load(BASE / f'g3/fold_{fold}/predictions.npz', allow_pickle=False) as v:
                rows = v['global_row']
                guard.admit(np.unique(v['dataset_id']), purpose='G3 single score release')
                np.testing.assert_array_equal(metadata['dataset_id'][rows], v['dataset_id'])
                assert {guard.split.assignment(int(s))[1] for s in np.unique(v['dataset_id'])} == {fold}
                for name, key in [('M0', 'M0'), ('G3_PAIRWISE_FULL', 'prediction')]:
                    value = float(ts_auc(metadata['target'][rows], v[key], metadata['time_online'][rows]))
                    scores[name][str(fold)] = value
                    records.append({'method': name, 'fold': fold, 'TS_AUC': value,
                        'series': len(np.unique(v['dataset_id'])), 'rows': len(rows),
                        'strict_nested_correction': name != 'M0', 'no_outer_label_fit': True})
        baseline = np.array([scores['M0'][str(f)] for f in range(5)])
        np.testing.assert_array_equal(baseline, [read_json(BASE / 'G1_OUTER_SCORES.json')['scores']['M0'][str(f)] for f in range(5)])
        summaries = []
        for method in scores:
            s = np.array([scores[method][str(f)] for f in range(5)])
            delta = s - baseline
            summaries.append({'method': method, 'folds': 5, 'mean_TS_AUC': float(s.mean()),
                'median_TS_AUC': float(np.median(s)), 'worst_TS_AUC': float(s.min()),
                'mean_delta_M0': float(delta.mean()), 'median_delta_M0': float(np.median(delta)),
                'worst_fold_delta_M0': float(delta.min()), 'positive_matched_folds': int((delta > 0).sum()),
                'conditional_DEVELOPMENT_only': True})
        for r in records:
            r['delta_M0'] = r['TS_AUC'] - baseline[r['fold']]
        csv_write('CV_RESULTS_NEXT7_G3.csv', records)
        csv_write('MODEL_COMPARISON_NEXT7_G3.csv', summaries)
        save('G3_OUTER_SCORES.json', {'created_utc': utc_now(), 'scores': scores, 'summaries': summaries,
            'M0_frozen_parity': 'PASS', 'new_full_scientific_candidates': 1})
        candidate = next(r for r in summaries if r['method'] == 'G3_PAIRWISE_FULL')
        save('G3_BOOTSTRAP_CANDIDATE_FROZEN.json', {'status': 'FROZEN', 'created_utc': utc_now(),
            'selected': ['G3_PAIRWISE_FULL'], 'draws': 5000 if candidate['mean_delta_M0'] >= .0025 else 2000,
            'references': ['M0', 'G2_FULL'], 'draw_count_decided_before_bootstrap': True,
            'no_post_interval_extension': True, 'not_a_promotion': True})
        usage = read_json(BASE / 'runtime/CANDIDATE_USAGE.json')
        usage.update(full_completed=15, G3_scored=['G3_PAIRWISE_FULL'], updated_utc=utc_now())
        save('runtime/CANDIDATE_USAGE.json', usage, False)
        print(summaries, flush=True)
    guard.check_state()
    note('G3 scores frozen; learned recipe budget exhausted', 'Continue uncertainty, conditional H calibration and runtime analysis', 'F6_TIME_MATCHED_RANK_CORRECTION')


if __name__ == '__main__':
    main()
