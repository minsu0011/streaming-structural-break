"""Check exported DEV tables against the closed research checkpoint; no rescoring."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.artifacts import sha256, utc_now, write_json


def main():
    state_path = ROOT / 'artifacts/real/RESEARCH_STATE.json'
    state = json.loads(state_path.read_text())
    reference = {row['experiment_id']: row for row in state['results']}
    checks, files = {}, {}

    def table(name):
        frame = pd.read_csv(ROOT / name, float_precision='round_trip')
        files[name] = {'sha256': sha256(ROOT / name), 'rows': len(frame)}
        checks[name + ':known_candidates'] = set(frame.experiment_id) <= set(reference)
        return frame

    models = table('MODEL_RESULTS_REAL.csv')
    cv = table('CV_RESULTS_REAL.csv')
    checks['model_table_exact_coverage'] = len(models) == len(reference) and set(models.experiment_id) == set(reference)
    checks['cv_exact_coverage'] = len(cv) == 5 * len(reference) and set(cv.experiment_id) == set(reference)
    checks['cv_keys_unique'] = not cv.duplicated(['experiment_id', 'fold']).any()
    checks['model_keys_unique'] = not models.duplicated('experiment_id').any()
    checks['complete_status_only'] = set(models.status) == {'COMPLETE'}
    maximum = 0.
    for eid, row in reference.items():
        actual = cv.loc[cv.experiment_id == eid].sort_values('fold')
        if actual.fold.tolist() != list(range(5)):
            raise RuntimeError('A candidate lacks the five frozen folds')
        maximum = max(maximum, float(np.max(np.abs(actual.ts_auc.to_numpy() - row['fold_scores']))))
        model_row = models.loc[models.experiment_id == eid].iloc[0]
        for key in ('mean', 'median', 'std', 'worst_fold', 'best_fold'):
            maximum = max(maximum, abs(float(model_row[key]) - row[key]))
    checks['all_exported_scores_match_checkpoint'] = maximum < 1e-12
    stored_fractures = pd.DataFrame(state['fractures'])
    profile_coverage = {}
    for name, slices, profile_key in (
        ('TIME_FRACTURE_REAL.csv', ['0-9','10-24','25-49','50-99','100-199','200-399','400+'], 'time_profile'),
        ('BREAK_AGE_DIAGNOSTICS.csv', ['0-4','5-9','10-19','20-49','50+'], 'age_profile'),
    ):
        frame = table(name)
        checks[name + ':coverage'] = len(frame) == 5 * len(slices) * len(reference) and set(frame.experiment_id) == set(reference)
        checks[name + ':keys_unique'] = not frame.duplicated(['experiment_id','fold','slice']).any()
        checks[name + ':slice_names'] = set(frame['slice']) == set(slices)
        checks[name + ':fold_names'] = set(frame.fold) == set(range(5))
        kind = 'online_time' if profile_key == 'time_profile' else 'evidence_age_with_negative_controls'
        keys = ['experiment_id','fold','slice']
        expected = stored_fractures.loc[stored_fractures.kind == kind].sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(frame.sort_values(keys).reset_index(drop=True), expected,
                                      check_dtype=False, check_like=True, check_exact=False, rtol=1e-12, atol=1e-12)
        checks[name + ':all_stored_fracture_rows_match'] = True
        maximum_profile = 0.
        for (eid, name_slice), values in frame.groupby(['experiment_id','slice'], sort=False):
            if profile_key in reference[eid]:
                maximum_profile = max(maximum_profile, abs(float(values.ts_auc.mean()) - reference[eid][profile_key][name_slice]))
        profile_coverage[name] = {'candidates_with_nested_summary': sum(profile_key in row for row in reference.values()),
                                 'candidates_with_full_per_fold_fracture_rows': len(reference)}
        checks[name + ':profile_matches'] = maximum_profile < 1e-12
    for name in ('BASELINE_RESULTS_REAL.csv','FEATURE_ABLATIONS_REAL.csv','SAMPLING_WEIGHTING_RESULTS.csv'):
        frame = table(name)
        checks[name + ':unique_complete'] = not frame.experiment_id.duplicated().any() and set(frame.status) == {'COMPLETE'}
        checks[name + ':summaries_match'] = all(abs(float(row['mean']) - reference[row['experiment_id']]['mean']) < 1e-12 for row in frame.to_dict('records'))
    report = {'status': 'PASS' if all(checks.values()) else 'FAIL', 'timestamp': utc_now(), 'checks': checks,
              'research_state_sha256': sha256(state_path), 'completed_candidates': len(reference),
              'max_export_score_difference': maximum, 'files': files,
              'profile_summary_coverage': profile_coverage,
              'scope': 'CSV identity, key coverage and numerical agreement with the closed DEV research checkpoint. No prediction, target loading, training or metric recomputation.',
              'LOCAL_FINAL_SEAL_OPENED': False, 'reduced_targets_used': False}
    write_json(ROOT / 'artifacts/validation/RESULT_EXPORT_AUDIT_REAL.json', report)
    print(report['status'], len(checks), 'export checks; failures:', [key for key,value in checks.items() if not value])
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
