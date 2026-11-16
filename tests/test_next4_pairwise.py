import numpy as np
import pytest
from scipy.optimize import check_grad
from src.next4 import pairwise as p


def settings(kind):
    return {'version':'honest_same_time_pairs_v1','kind':kind,'pairs':1048576,'seed':20260908,'temperature':.1,'ridge':.01,'maxiter':200}


@pytest.mark.parametrize('kind',p.KINDS)
def test_one_hundred_pure_current_point_prefix_cases(kind):
    rng=np.random.default_rng(20260908)
    for case in range(100):
        a=rng.uniform(size=(65,2));b=a.copy();prefix=1+case%48;b[prefix:]=rng.uniform(size=(65-prefix,2))
        left=p.features(a,kind);right=p.features(b,kind)
        np.testing.assert_array_equal(left[:prefix],right[:prefix])
        weights=rng.normal(size=left.shape[1]);model=p.PairwiseModel(settings(kind),weights,p.source_hash(),{})
        np.testing.assert_array_equal(model.predict(a),np.array([model.predict_one(*row) for row in a],dtype=np.float32))


def test_same_time_pair_mass_and_reproducible_coordinates():
    y=np.array([0,1,0,0,1,1,1,0,1]);t=np.array([0,0,1,1,1,1,1,2,3]);ids=np.arange(9)
    a,b,proof=p.sample_pairs(y,t,ids,10000,20);aa,bb,_=p.sample_pairs(y,t,ids,10000,20)
    np.testing.assert_array_equal(a,aa);np.testing.assert_array_equal(b,bb)
    np.testing.assert_array_equal(t[a],t[b]);assert np.all(y[a]==1) and np.all(y[b]==0)
    assert proof['eligible_pair_mass']==7 and proof['eligible_times']==2
    # The 1:6 time-pair-mass allocation should agree within five standard errors.
    assert abs(np.mean(t[a]==0)-1/7)<5*np.sqrt((1/7)*(6/7)/10000)


def test_pair_loss_gradient_independent_finite_difference():
    rng=np.random.default_rng(22);x=rng.normal(size=(100,6));w=rng.normal(size=6);anchor=np.r_[.8,.2,np.zeros(4)]
    error=check_grad(lambda v:p.loss_gradient(v,x,anchor,.1,.01)[0],lambda v:p.loss_gradient(v,x,anchor,.1,.01)[1],w)
    assert error<1e-6


@pytest.mark.parametrize('kind',p.KINDS)
def test_small_honest_fit_save_load_and_pair_loss_decrease(tmp_path,kind):
    rng=np.random.default_rng(34);ids=np.tile(np.arange(40),10);times=np.repeat(np.arange(10),40)
    target=rng.integers(0,2,len(ids));scores=np.column_stack([.2+.5*target+rng.uniform(0,.15,len(ids)),rng.uniform(size=len(ids))])
    model,detail=p.fit(scores,target,times,ids,settings(kind));path=tmp_path/'model.joblib';model.save(path)
    np.testing.assert_array_equal(model.predict(scores),p.PairwiseModel.load(path).predict(scores))
    assert detail['final_pair_loss']<=detail['initial_pair_loss'] and detail['new_base_fits']==0
    assert np.isfinite(model.predict(scores)).all()


@pytest.mark.parametrize('bad',[np.ones((4,3)),np.full((4,2),np.nan),np.full((4,2),2)])
def test_no_extra_or_invalid_meta_inputs(bad):
    with pytest.raises(ValueError):p.features(bad,'linear')


def test_invalid_pair_pool_rejected():
    with pytest.raises(ValueError):p.sample_pairs(np.ones(3),np.zeros(3),np.arange(3),100,1)
    with pytest.raises(ValueError):p.sample_pairs(np.array([0,1]),np.zeros(2),np.ones(2),100,1)
