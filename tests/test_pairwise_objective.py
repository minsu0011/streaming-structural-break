import numpy as np
from src.models.pairwise import WithinTimePairObjective,fit_pairwise
from src.models.learners import ModelBundle

def test_ranknet_derivatives_and_same_time_pair_contract():
    y=np.array([0,1,0,1,0,1],dtype=np.uint8);t=np.array([0,0,1,1,2,2])
    objective=WithinTimePairObjective.build(y,t,pairs_per_positive=8)
    assert np.array_equal(t[objective.positive],t[objective.negative])
    assert np.all(y[objective.positive]==1) and np.all(y[objective.negative]==0)
    p=np.linspace(-1,1,len(y));g,h=objective(y,p);epsilon=1e-4
    for j in range(len(p)):
        delta=np.zeros(len(p));delta[j]=epsilon
        numerical_g=(objective.loss(p+delta)-objective.loss(p-delta))/(2*epsilon)
        numerical_h=(objective.loss(p+delta)-2*objective.loss(p)+objective.loss(p-delta))/epsilon**2
        np.testing.assert_allclose(g[j],numerical_g,rtol=0,atol=1e-8)
        np.testing.assert_allclose(h[j],numerical_h,rtol=0,atol=1e-6)
    assert abs(g.sum())<1e-12
    np.testing.assert_allclose(objective.loss(p+10),objective.loss(p),rtol=0,atol=1e-12)

def test_pairwise_compact_serialization(tmp_path):
    rng=np.random.default_rng(141);x=rng.normal(size=(600,57)).astype(np.float32);t=np.tile(np.arange(20),30);y=(x[:,0]+x[:,1]>.1).astype('uint8')
    model,meta=fit_pairwise(x,y,t,params={'n_estimators':15})
    assert meta['pair_count']<=int(y.sum())*8 and not meta['validation_labels_used']
    path=tmp_path/'rank.joblib';model.save(path);loaded=ModelBundle.load(path)
    np.testing.assert_array_equal(model.predict(x),loaded.predict(x))
