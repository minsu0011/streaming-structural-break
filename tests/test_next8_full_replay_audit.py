"""Fresh scalar bank also carries H-seeded histories without label access."""
import src.next8
import numpy as np
import pytest
from src.next8.features import FeatureState
from src.next8.full_replay_audit import FreshSeriesCache


def test_full_scalar_bank_matches_batch_and_seeded_lags():
    rng=np.random.default_rng(8101629);h=rng.normal(size=512).astype(np.float32);o=rng.normal(size=73).astype(np.float32)
    reference=FeatureState(h);seed=reference.history.copy();values,_=reference.replay(o)
    fresh=FreshSeriesCache(123,h,o)
    np.testing.assert_array_equal(fresh.values.view(np.uint32),values.view(np.uint32))
    np.testing.assert_array_equal(fresh.lag(np.array([0]),[0,64,93,117],1)[0],seed[-1,[0,64,93,117]])
    np.testing.assert_array_equal(fresh.lag(np.array([20]),[0,64,93,117],16)[0],values[4,[0,64,93,117]])
    with pytest.raises(AssertionError,match='labels'):np.asarray(fresh.meta['target'])


def test_short_H_fresh_audit_has_explicit_unknown_states():
    fresh=FreshSeriesCache(123,np.zeros(100,dtype=np.float32),np.ones(8,dtype=np.float32))
    assert np.all(fresh.values==0) and np.all(fresh.states(np.arange(8),[0,64,93,117])==3)
