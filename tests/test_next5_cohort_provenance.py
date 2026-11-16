import pytest
from src.next5.artifacts import BASE
from src.next5.cohort_provenance import rows, annotate_paired_alarms


def test_actual_fixed_fold_null_cohorts_are_separated_without_changing_values():
    path = BASE / 'synthetic/candidates/X2_SCALE_TAIL/full/fold_0/ANALYSIS/PAIRED_ALARMS.csv'
    original = rows(path)
    annotated = annotate_paired_alarms(path, original)
    assert [{key: row[key] for key in original[0]} for row in annotated] == original
    collisions = [row for row in annotated if row['scenario'] == 'N0' and row['reference'] == 'BL80'
        and int(row['horizon']) == 1024 and float(row['alpha']) == .05 and row['event'] == 'null_eventual']
    assert len(collisions) == 2
    assert {r['cohort_kind'] for r in collisions} == {'counterexample', 'long_null'}
    assert len({r['candidate_cohort_prediction_sha256'] for r in collisions}) == 2


def test_reordered_scalar_rows_cannot_be_assigned_by_score_values():
    path = BASE / 'synthetic/candidates/X2_SCALE_TAIL/full/fold_0/ANALYSIS/PAIRED_ALARMS.csv'
    original = rows(path)
    with pytest.raises(ValueError, match='row order|coordinate'):
        annotate_paired_alarms(path, list(reversed(original)))
