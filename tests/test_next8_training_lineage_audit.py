"""Validation envelopes and late-only training cannot pass through metadata."""
import src.next8
import pytest
from src.next8.data import FitScope
from src.next8.training_lineage_audit import validate_training_record


def test_independent_lineage_rejects_wrong_group_or_late_scope():
    scope = FitScope(frozenset({1, 2}), frozenset({3}), frozenset({3, 4}), 'outer_0_EARLY_T199')
    scopes = {scope.name: scope}
    assert validate_training_record(scope.provenance(), scopes) == scope.provenance()
    with pytest.raises(ValueError, match='training_ids_sha256'):
        validate_training_record({**scope.provenance(), 'training_ids_sha256': 'wrong_group'}, scopes)
    with pytest.raises(ValueError, match='Unregistered'):
        validate_training_record({**scope.provenance(), 'scope': 'ALL_DEV'}, scopes)
    with pytest.raises(ValueError, match='beyond199'):
        validate_training_record({**scope.provenance(), 'training_max_online_time': 400}, scopes)
