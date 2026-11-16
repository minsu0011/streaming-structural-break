from src.next6_extension.categorical_controls import generated
from src.next6_extension.factorized import discover_and_fit,evaluate_pair

def fit(kind):
    a,b,p,ids=generated(256,64,[20260913,88],.9,'uniform',kind);groups=set(map(int,ids));return discover_and_fit(a,b,p,ids,groups,{g:g for g in groups})

def test_known_changed_relationship_is_discovered():
    g,_=fit('changed_link');assert any(e['source']==0 and e['target']==4 for e in g.selected)

def test_stable_predictive_relationship_is_not_changed_relationship():
    g,_=fit('stable_link');assert not any(e['source']==0 and e['target']==4 for e in g.selected)

def test_large_marginal_shift_is_not_relationship_interaction():
    g,_=fit('marginal_shift');assert not g.selected
