import json
from pathlib import Path
import numpy as np
import pytest
from src.next.guard import SealGuard, SealAccessError

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def guard():
    return SealGuard(ROOT)


def test_next_seal_membership_is_exact(guard):
    assert len(guard.seal_ids) == 2000
    assert len(guard.admit()) == 8000
    assert not guard.seal_ids & guard.dev_ids


@pytest.mark.parametrize('purpose', ['training', 'validation', 'OOF', 'prediction', 'experiment ledger', 'mechanism diagnostics'])
def test_next_seal_operation_fails(guard, purpose):
    with pytest.raises(SealAccessError):
        guard.admit([min(guard.dev_ids), min(guard.seal_ids)], purpose=purpose)


def test_next_seal_labels_fail_before_parquet_open(guard, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('A seal label request reached the data reader')
    monkeypatch.setattr('src.next.guard.ds.dataset', forbidden)
    with pytest.raises(SealAccessError):
        guard.labels([min(guard.seal_ids)])


def test_next_reduced_read_forbidden(guard, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('A reduced request reached the reader')
    monkeypatch.setattr('src.next.guard.ds.dataset', forbidden)
    with pytest.raises(SealAccessError):
        guard.scanner('X_test.reduced.parquet')


def test_next_unknown_and_nonintegral_ids_fail(guard):
    for ids in [[-1], [True], [float(min(guard.dev_ids))], []]:
        with pytest.raises(SealAccessError):
            guard.admit(ids)


def test_next_overlap_and_seal_partition_fail(guard):
    a, b = sorted(guard.dev_ids)[:2]
    with pytest.raises(SealAccessError):
        guard.partition([a], [a])
    with pytest.raises(SealAccessError):
        guard.partition([a], [min(guard.seal_ids)])
    assert guard.partition([a], [b]) == ({a}, {b})


def test_next_oof_and_ledger_rows_fail(guard):
    with pytest.raises(SealAccessError):
        guard.record_rows(np.array([min(guard.seal_ids)]), purpose='experiment ledger')
