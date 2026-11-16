import numpy as np
from src.next6_extension.factorized_coherent import fit_pair,null_interaction_error,discover_and_fit
from src.next6_extension.categorical_controls import generated

def test_sparse_context_backoff_preserves_exact_null():
    for groups in (16,32,64,128):
        for seed in range(32):
            a,b,p,ids=generated(groups,32,[20260913,910,seed],.6,'central80percent','marginal_shift');full,null,support,receipt=fit_pair(a,b,p,ids,0,4)
            assert null_interaction_error(null)<1e-8
            for own in receipt['whole_ownstate_backoff']:
                np.testing.assert_array_equal(full[:,own],null[:,own]);assert np.all(support[:,own]==0)

def test_known_changed_relationship_remains_detectable():
    a,b,p,ids=generated(256,64,[20260913,88],.9,'uniform','changed_link');groups=set(map(int,ids));g,_=discover_and_fit(a,b,p,ids,groups,{s:s for s in groups});assert any(e['source']==0 and e['target']==4 for e in g.selected);g.validate()
