import numpy as np
import pytest
from src.next8.full_bank_gate_control import coordinate_values, full_bank_fixture, full_bank_names, SIGNALS


def test_full_bank_simulation_shares_targets_but_never_future_information():
    labels = np.zeros((40, 80), dtype=np.uint8)
    labels[:, 30:] = 1
    for j in (0, 6, 9, 11):
        original = coordinate_values(j, labels)
        changed = labels.copy()
        changed[:, 50:] = 1 - changed[:, 50:]
        np.testing.assert_array_equal(original[:, :50], coordinate_values(j, changed)[:, :50])
        np.testing.assert_array_equal(original[:, :50], coordinate_values(j, labels[:, :50]))
    np.testing.assert_array_equal(coordinate_values(11, labels), coordinate_values(11, 1 - labels))
    assert not np.array_equal(coordinate_values(11, labels), coordinate_values(15, labels))


def test_full153_fixture_keeps_real_detailed_menu_and_blocks_competition_loader(monkeypatch):
    import src.next8.cache as cache_module
    import src.next8.screening as screening
    from src.next8.features import DETAILED
    def forbidden():
        raise AssertionError('Artificial bank accessed competition loader')
    monkeypatch.setattr(cache_module, 'source', forbidden)
    fixture = full_bank_fixture(series=120, length=64)
    assert fixture.cache.values.shape == (120 * 64, 153)
    assert fixture.cache.series['seed_values'].shape == (120, 16, 153)
    assert len(fixture.coordinate_sha256) == 153
    assert set(SIGNALS) <= set(DETAILED)
    for scope in fixture.scopes:
        assert not scope.training_ids & scope.validation_ids
    assert set.union(*(set(s.validation_ids) for s in fixture.scopes)) == fixture.parent.training_ids
    assert min(fixture.parent.training_ids) == 800000
    before = screening.NAMES, screening.DETAILED
    with pytest.raises(RuntimeError):
        with full_bank_names():
            assert len(screening.NAMES) == 153 and screening.DETAILED == DETAILED
            raise RuntimeError('deliberate restoration probe')
    assert (screening.NAMES, screening.DETAILED) == before
