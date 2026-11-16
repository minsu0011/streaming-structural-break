import copy
from src.next9.certification import orientation_pass,tabular_comparison,domain_replication


def example():
    ci=lambda lo,hi:dict(status='SUPPORTED',ci_low=lo,ci_high=hi)
    return dict(status='COMPLETE',fold=0,method='O3_MOVE',difference_Holm_p=.01,
        evidence=dict(orientation=dict(first=ci(.01,.1),second=ci(-.1,-.01)),
            contrasts={f'O3_MOVE__vs__{ref}':dict(TS_AUC=ci(.001,.002),logloss_gain=ci(.001,.002))
                       for ref in ['O1','STATE_ONLY','O6_TABULAR']}))


def test_significant_vs_nonsignificant_does_not_certify():
    r=example();assert orientation_pass(r,1)
    r['evidence']['orientation']['second']['ci_high']=.01
    assert not orientation_pass(r)
    r=example();r['difference_Holm_p']=.051
    assert not orientation_pass(r)
    r=example();r['evidence']['contrasts']['O3_MOVE__vs__STATE_ONLY']['TS_AUC']['ci_low']=0
    assert not orientation_pass(r)


def test_nonsignificance_is_not_equivalence():
    r=example();c=r['evidence']['contrasts']['O3_MOVE__vs__O6_TABULAR']['TS_AUC']
    c['ci_low']=-.01;assert not tabular_comparison(r)['noninferior']
    c['ci_low']=-.0009;assert tabular_comparison(r)['noninferior'] and not tabular_comparison(r)['superior']


def test_domain_requires_replication_and_vetoes_opposite_pattern():
    a=example();b=copy.deepcopy(a);b['fold']=1
    assert not domain_replication([a],1)['supported']
    assert domain_replication([a,b],1)['supported']
    c=copy.deepcopy(a);c['fold']=2
    o=c['evidence']['orientation'];o['first'],o['second']=o['second'],o['first']
    assert not domain_replication([a,b,c],1)['supported']
