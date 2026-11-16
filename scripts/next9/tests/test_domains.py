import src.next9
from types import SimpleNamespace
import numpy as np
from src.next9.data import FitScope
from src.next9.domains import fit_domains, masks


def test_domain_cutpoints_use_only_training_series():
    context=np.zeros((8,6));context[:,2]=np.arange(8);context[:,4]=np.arange(8)*3
    bank=SimpleNamespace(ids=np.arange(8),series={'context':context,'H_length':np.array([1249,1250,1749,1750,2499,2500,3999,4000])},
        times=np.array([9,10,24,25,99,100,399,400]),series_index=np.arange(8))
    scope=FitScope(frozenset(range(6)),frozenset([6,7]),frozenset([6,7]),'toy')
    first=fit_domains(bank,scope)
    context[6:,2:]=1e20
    assert fit_domains(bank,scope)==first
    domains=list(masks(bank,np.arange(8),first))
    for family in ['TIME','H_LENGTH','H_LAG1','H_TAIL3']:
        np.testing.assert_array_equal(sum(m for f,n,m in domains if f==family),np.ones(8))
    assert dict((n,m) for f,n,m in domains)['TIME_6'][7]
    assert dict((n,m) for f,n,m in domains)['H_LENGTH_4'][7]


def test_degenerate_h_domains_remain_unknown():
    context=np.zeros((4,6));context[3,2]=np.nan
    bank=SimpleNamespace(ids=np.arange(4),series={'context':context,'H_length':np.full(4,1000)},
        times=np.arange(4),series_index=np.arange(4))
    scope=FitScope(frozenset([0,1,2]),frozenset([3]),frozenset([3]),'toy')
    spec=fit_domains(bank,scope)
    assert not spec['terciles']['H_LAG1']['available']
    for family,name,mask in masks(bank,np.arange(4),spec):
        if family in ['H_LAG1','H_TAIL3']:
            assert mask.all() if name.endswith('_3') else not mask.any()
