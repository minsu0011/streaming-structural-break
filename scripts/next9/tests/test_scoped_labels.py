import src.next9
from types import SimpleNamespace
import numpy as np
import pytest
from src.next9.data import FitScope
from src.next9.labels import map_stored_vector, validate_rows


def test_selective_stored_label_mapping(tmp_path):
    p = tmp_path/'fixture.npz'
    values = np.arange(1000, dtype=np.int16)
    np.savez(p, target=values, tau=np.array([99999]))
    mapped = map_stored_vector(p)
    np.testing.assert_array_equal(mapped[[1, 17, 991]], values[[1, 17, 991]])
    assert not mapped.flags.writeable


def test_no_outer_feedback():
    bank = SimpleNamespace(row_ids=np.repeat(np.arange(6), 3), times=np.tile([0, 199, 400], 6))
    scope = FitScope(frozenset([0,1]), frozenset([2,3]), frozenset([4,5]), 'toy')
    np.testing.assert_array_equal(validate_rows(bank, np.array([0,1]), scope, 'TRAIN', early=True), [0,1])
    with pytest.raises(ValueError, match='late labels'):
        validate_rows(bank, np.array([2]), scope, 'TRAIN', early=True)
    with pytest.raises(ValueError, match='outside'):
        validate_rows(bank, np.array([12]), scope, 'TRAIN')
    with pytest.raises(ValueError, match='globally frozen'):
        validate_rows(bank, np.array([6]), scope, 'FROZEN_OUTER_EVALUATION')
    receipt = dict(status='FROZEN', scope=scope.provenance())
    np.testing.assert_array_equal(validate_rows(bank, np.array([6]), scope, 'FROZEN_OUTER_EVALUATION', prediction_receipt=receipt), [6])


def test_internal_validation_cannot_alias_outer():
    bank = SimpleNamespace(row_ids=np.arange(4), times=np.zeros(4))
    scope = FitScope(frozenset([0,1]), frozenset([2,3]), frozenset([2,3]), 'outer')
    with pytest.raises(ValueError, match='Outer labels'):
        validate_rows(bank, np.array([2]), scope, 'INTERNAL_VALIDATION')
