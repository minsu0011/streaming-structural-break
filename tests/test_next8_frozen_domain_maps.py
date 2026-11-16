"""Heldout scoring reads an already frozen H map and performs no new fit."""
import src.next8
from types import SimpleNamespace
import numpy as np
from tests.test_next8_movement_parents import feature_fixture
from src.next8.h_context import HStateSpec,CONTEXT_MENU
from src.next8.evaluation import domains


def test_scoring_H_maps_do_not_refit_on_current_population(monkeypatch):
    c,scope=feature_fixture();c.series['dataset_id']=np.arange(256)
    c.data=SimpleNamespace(series={'historical_offsets':np.r_[0,np.cumsum(c.series['H_length'])]})
    specs={k:vars(HStateSpec.fit(c,scope,k)) for k in CONTEXT_MENU}
    rows=np.flatnonzero(c.meta['dataset_id']>=128)
    expected=[(family,name,mask.copy()) for family,name,mask,_ in domains(c,scope,rows,specs)]
    c.series['context'][:128]*=100
    def fail(*a,**kw):raise AssertionError('Scoring tried to fit H domains')
    monkeypatch.setattr(HStateSpec,'fit',fail)
    actual=list(domains(c,scope,rows,specs))
    for (family,name,mask),(ef,en,em) in zip([(a,b,c) for a,b,c,_ in actual],expected):
        assert (family,name)==(ef,en);np.testing.assert_array_equal(mask,em)
