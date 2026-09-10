import numpy as np
import pytest
from src.next.variance_evidence import VarianceState,VarianceConfig
from src.next.serial_variance_evidence import SerialVarianceState,SerialVarianceConfig
from src.next.channel_evidence import CHANNEL_STATS
from src.next2.pruned_runtime import CompactPairState,LOCAL_BAYES_STATS


@pytest.mark.parametrize('kind',['full','local_bayes','serial'])
@pytest.mark.parametrize('historical_size',[128,2048])
def test_compact_retained_features_match_original_across_age_wrap(kind,historical_size):
    rng = np.random.default_rng(817)
    h = rng.standard_t(5,historical_size).astype(np.float32)
    o = rng.normal(size=1701).astype(np.float32)
    o[300:600] *= .4
    o[800] = 40
    settings = {'order':8,'normalization':'arch1','max_age':512}
    original = SerialVarianceState(h,SerialVarianceConfig(**settings)) if kind=='serial' else VarianceState(h,VarianceConfig(**settings))
    compact = CompactPairState(h,settings,kind)
    columns = [c*13+CHANNEL_STATS.index(s) for c in range(2) for s in LOCAL_BAYES_STATS] if kind=='local_bayes' else list(range(26))
    left = original.replay(o)[:,columns]
    right = np.stack([compact.update(x).copy() for x in o])
    np.testing.assert_array_equal(left,right)
    assert compact.evidence_args[5].shape==(513,2)
    if kind=='local_bayes':
        assert len(compact.evidence_args)==8


def test_compact_prefix_has_no_future_suffix_dependence():
    rng = np.random.default_rng(901)
    h,o = rng.normal(size=2048),rng.normal(size=1500)
    settings = {'order':8,'normalization':'arch1','max_age':512}
    state = CompactPairState(h,settings,'local_bayes')
    first = np.stack([state.update(x).copy() for x in o[:700]])
    state = CompactPairState(h,settings,'local_bayes')
    whole = np.stack([state.update(x).copy() for x in np.r_[o[:700],np.full(800,1e6)]])
    np.testing.assert_array_equal(first,whole[:700])
