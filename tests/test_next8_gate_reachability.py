import numpy as np
import pytest
from src.next8.gate_reachability import coordinate, controlled_cache, single_control_coordinate, WORLDS, cases


@pytest.mark.parametrize('world', WORLDS)
def test_control_simulator_future_regime_and_observations_cannot_change_prefix(world):
    rng = np.random.default_rng(76)
    noise = rng.normal(size=(40, 80))
    labels = np.zeros(noise.shape, dtype=np.uint8)
    labels[:, 30:] = 1
    signs = np.where(rng.random(noise.shape) > .5, 1, -1)
    original = coordinate(noise, labels, signs, world)
    noise[:, 50:] *= -7
    labels[:, 50:] = 1 - labels[:, 50:]
    changed = coordinate(noise, labels, signs, world)
    np.testing.assert_array_equal(original[:, :50], changed[:, :50])
    np.testing.assert_array_equal(original[:, :50], coordinate(noise[:, :50], labels[:, :50], signs[:, :50], world))


def test_control_cache_never_initializes_competition_loader_and_memberships_are_separate(monkeypatch):
    import src.next8.cache as cache_module
    import src.next8.screening as screening
    def forbidden():
        raise AssertionError('Statistical control touched real data loader')
    monkeypatch.setattr(cache_module, 'source', forbidden)
    fixture = controlled_cache('OWN_STATE_REVERSAL', .9, 8101611, series=120, length=64)
    assert set(fixture.cache.meta) == {'dataset_id', 'time_online', 'target'}
    assert fixture.cache.meta['dataset_id'].min() == 500000
    held = [set(s.validation_ids) for s in fixture.scopes]
    assert set.union(*held) == fixture.parent.training_ids
    assert all(not held[a] & held[b] for a in range(3) for b in range(a))
    before = screening.NAMES, screening.DETAILED
    with pytest.raises(RuntimeError):
        with single_control_coordinate('NULL'):
            assert len(screening.NAMES) == 1
            raise RuntimeError('deliberate isolated-context failure')
    assert (screening.NAMES, screening.DETAILED) == before
    assert len(cases()) == len(set(cases())) == 16
