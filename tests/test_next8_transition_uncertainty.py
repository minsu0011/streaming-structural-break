"""Per-series normalized group counts reproduce the actual transition model."""
import src.next8
import numpy as np
from src.next8.transition_uncertainty import group_transition_counts,table_uncertainty
from src.next8.transitions import FactorizedTransition
from src.next8.data import FitScope


def test_group_weighted_table_parity_and_unknown_exclusion():
    ids=np.repeat(np.arange(64),np.arange(64)%3+7);own=np.arange(len(ids))%3;target=(np.arange(len(ids))//2)%3
    own[::17]=3
    scope=FitScope(frozenset(range(64)),frozenset([64]),frozenset([64]),'toy')
    model=FactorizedTransition.fit(own,np.empty((len(ids),0),dtype=int),target,ids,[],scope)
    groups,counts=group_transition_counts(own,target,ids)
    np.testing.assert_allclose(counts.sum(axis=1),1.,atol=1e-15)
    r=table_uncertainty(counts)
    np.testing.assert_allclose(r['point'],model.own_table,atol=1e-14)
    assert len(groups)==64 and np.all(r['low']<=r['high']) and np.all(r['low']>=0) and np.all(r['high']<=1)
