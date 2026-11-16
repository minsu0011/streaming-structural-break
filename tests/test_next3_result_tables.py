import csv
import io
import pytest
from src.next3.result_tables import append_history, close_recipe, inherited_validation, selected_identity_pair


def test_history_preserves_bom_crlf_and_quoted_embedded_newline():
    before = b'\xef\xbb\xbfexperiment_id,reason,new_fits\r\nOLD,"first\r\nsecond, line",2\r\n'
    after = append_history(before, [{'experiment_id': 'NEW', 'reason': 'closed, failed', 'new_fits': 7}])
    assert after[:len(before)] == before
    rows = list(csv.DictReader(io.StringIO(after.decode('utf-8-sig'))))
    assert rows == [{'experiment_id': 'OLD', 'reason': 'first\r\nsecond, line', 'new_fits': '2'},
                    {'experiment_id': 'NEW', 'reason': 'closed, failed', 'new_fits': '7'}]


@pytest.mark.parametrize('rows', [
    [{'experiment_id': 'OLD'}], [{'experiment_id': 'NEW'}, {'experiment_id': 'NEW'}],
    [{'experiment_id': 'NEW', 'unknown': 'would alter header'}],
])
def test_history_rejects_ambiguous_append(rows):
    with pytest.raises(ValueError): append_history(b'experiment_id,reason\nOLD,prior\n', rows)


def test_history_rejects_unterminated_old_record():
    with pytest.raises(ValueError): append_history(b'experiment_id,reason\nOLD,prior', [{'experiment_id': 'NEW'}])


def test_identical_selection_has_zero_strict_positive_probability_without_fake_draws():
    row = selected_identity_pair({'experiment_id': 'BL', 'full': {'mean': .623}})
    assert row['candidate'] == row['reference'] == 'BL'
    assert row['ci_2_5'] == row['ci_97_5'] == row['probability_delta_positive'] == 0
    assert row['replicates'] == row['new_draws'] == 0 and row['analytic_identity']


def recipe():
    return {'experiment_id': 'NEW', 'scientific_role_selected': False, 'disposition': 'FULL_CV_NO_PROMOTION',
        'stages': {'full': {'mean': .6, 'fold_scores': [.58, .59, .6, .61, .62]}},
        'reason': 'Mean gain below fixed gate', 'recipe_evidence_binding_sha256': 'a'*64,
        'not_evaluated_after_rejection': {'PERFORMANCE': ['ALT/H']}}


def test_closed_full_admission_is_not_a_final_acceptance():
    row = close_recipe({'experiment_id': 'NEW', 'status': 'FULL_VALIDATION_ELIGIBLE', 'full_mean': '.6'}, recipe())
    assert row['status'] == 'FULL_CV_NO_PROMOTION'
    assert row['original_stage_status'] == 'FULL_VALIDATION_ELIGIBLE'
    assert row['scientific_role_selected'] is False and row['promotion'] is False


@pytest.mark.parametrize('mutation', ['promotion', 'wrong_id', 'wrong_mean', 'screen_with_full'])
def test_closed_recipe_rejects_evidence_conflation(mutation):
    record = recipe(); row = {'experiment_id': 'NEW', 'status': 'COMPLETE', 'full_mean': '.6'}
    if mutation == 'promotion': record['scientific_role_selected'] = True
    elif mutation == 'wrong_id': row['experiment_id'] = 'OTHER'
    elif mutation == 'wrong_mean': row['full_mean'] = '.61'
    else: record['disposition'] = 'SCREEN_FAIL_STOP'
    with pytest.raises(ValueError): close_recipe(row, record)


def heldout_rows():
    return [{'candidate': 'BL', 'reference': 'J', 'split': 'LEAVE_ONE_H_REGIME_OUT', 'fold': str(fold),
             'record_type': 'REFITTED_LEAVE_H_REGIME_OUT', 'ts_auc': str(score), 'mean': '.6', 'worst': '.58'}
            for fold, score in enumerate([.58, .59, .6, .61, .62])]


def test_inherited_H_retains_historical_reference_and_excludes_original_OOF_slice():
    rows = heldout_rows()+[{**heldout_rows()[0], 'record_type': 'ORIGINAL_OOF_H_REGIME_SLICE', 'ts_auc': '.9'}]
    selected = inherited_validation(rows, role_names={'BL'}, expected_kind='REFITTED_LEAVE_H_REGIME_OUT',
        source='OLD.csv', sha='b'*64)
    assert len(selected) == 5 and {row['reference'] for row in selected} == {'J'}
    assert all(row['new_fits_in_NEXT3'] == 0 and not row['new_independent_validation'] for row in selected)
    assert all(row['validation_kind'] == 'REFITTED_LEAVE_H_REGIME_OUT' for row in selected)


@pytest.mark.parametrize('mutation', ['missing_fold', 'duplicate_fold', 'wrong_summary'])
def test_inherited_H_requires_complete_and_consistent_folds(mutation):
    rows = heldout_rows()
    if mutation == 'missing_fold': rows.pop()
    elif mutation == 'duplicate_fold': rows[-1]['fold'] = '0'
    else: rows[0]['mean'] = '.7'
    with pytest.raises(ValueError):
        inherited_validation(rows, role_names={'BL'}, expected_kind='REFITTED_LEAVE_H_REGIME_OUT',
                             source='OLD.csv', sha='b'*64)
