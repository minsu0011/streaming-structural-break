import numpy as np
import pytest
from scipy.special import expit
from src.next.ranking_objectives import HistogramPairObjective


def exact_pairs(y,p,t):
    gradient=np.zeros(len(y));hessian=np.zeros(len(y))
    total=0
    for step in np.unique(t):
        positive=np.flatnonzero((t==step)&(y==1))
        negative=np.flatnonzero((t==step)&(y==0))
        for a in positive:
            for b in negative:
                q=expit(-(p[a]-p[b]));v=q*(1-q)
                gradient[a]-=q;gradient[b]+=q
                hessian[a]+=v;hessian[b]+=v
                total+=1
    scale=len(y)/(2*total)
    return gradient*scale,np.maximum(hessian*scale,1e-12)


def test_histogram_pair_gradient_is_exact_when_margin_cells_do_not_collide():
    rng=np.random.default_rng(928)
    times=np.repeat(np.arange(5),12)
    target=np.tile([0,1]*6,5).astype(np.uint8)
    prediction=np.tile(np.arange(12)/6,5).astype(float)
    obj=HistogramPairObjective.build(target,times,bins=64)
    actual=obj(target,prediction)
    expected=exact_pairs(target,prediction,times)
    for a,b in zip(actual,expected):
        np.testing.assert_allclose(a,b,rtol=0,atol=1e-14)
    assert abs(actual[0].sum())<1e-12


def test_histogram_pair_gradient_exact_at_common_initial_margin():
    times=np.repeat(np.arange(4),9)
    target=np.tile([0,0,1],12).astype(np.uint8)
    prediction=np.zeros(len(target))
    obj=HistogramPairObjective.build(target,times,bins=128)
    for a,b in zip(obj(target,prediction),exact_pairs(target,prediction,times)):
        np.testing.assert_allclose(a,b,rtol=0,atol=1e-14)


def test_histogram_pair_approximation_has_small_error_and_no_cross_time_pairs():
    rng=np.random.default_rng(81)
    times=np.repeat(np.arange(12),35)
    target=rng.integers(0,2,len(times),dtype=np.uint8)
    prediction=rng.normal(size=len(times))
    obj=HistogramPairObjective.build(target,times,bins=128)
    g,h=obj(target,prediction)
    eg,eh=exact_pairs(target,prediction,times)
    assert np.max(np.abs(g-eg))<.004
    assert np.max(np.abs(h-eh))<.004
    shifted=prediction+times*3.
    sg,sh=obj(target,shifted)
    np.testing.assert_allclose(g,sg,atol=1e-14,rtol=0)
    np.testing.assert_allclose(h,sh,atol=1e-14,rtol=0)


def test_histogram_objective_rejects_changed_training_label_order():
    y=np.array([0,1,0,1],dtype=np.uint8)
    obj=HistogramPairObjective.build(y,np.zeros(4,dtype=int))
    with pytest.raises(RuntimeError):
        obj(y[::-1],np.zeros(4))


@pytest.mark.parametrize('objective',[{'kind':'binary_query_sorted'},{'kind':'histogram_ranknet','bins':64},
    {'kind':'sampled_ranknet','pairs_per_positive':8},{'kind':'rank_xendcg'},{'kind':'lambdarank'}])
def test_ranking_native_export_and_reload_on_small_training_queries(objective,tmp_path):
    from src.next.ranking_objectives import fit_ranking
    from src.next.engine import EngineConfig,FEATURE_NAMES
    from src.next.predictor import NextBundle
    class Guard:
        def partition(self,a,b):
            assert not set(a)&set(b)
    rng=np.random.default_rng(159)
    ids=np.repeat(np.arange(32),6)
    t=np.tile(np.arange(6),32)
    x=rng.normal(size=(len(ids),3)).astype(np.float32)
    y=(x[:,0]+.4*x[:,1]>0).astype(np.uint8)
    meta={'dataset_id':ids,'time_online':t,'target':y}
    train,valid=np.flatnonzero(ids<24),np.flatnonzero(ids>=24)
    model,prediction,details=fit_ranking(x,meta,train,valid,guard=Guard(),config=EngineConfig(),names=FEATURE_NAMES[:3],objective=objective)
    assert np.isfinite(prediction).all()
    assert np.all((prediction>=0)&(prediction<=1))
    path=tmp_path/'model.joblib'
    model.save(path)
    loaded=NextBundle.load(path)
    np.testing.assert_array_equal(loaded.predict(x[valid]).astype(np.float32),prediction)
    assert path.stat().st_size<2_000_000
