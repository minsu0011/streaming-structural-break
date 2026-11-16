import numpy as np
import pytest
from scipy.optimize import check_grad
from src.next4 import state_readout as r
from src.next4 import temporal_state


SETTINGS={'version':'coupled_linear_state_v1','kind':'two_pole_readout'}


def test_weighted_loss_gradient_finite_difference():
    rng=np.random.default_rng(1);x=rng.normal(size=(90,32));y=rng.integers(0,2,90);weights=rng.uniform(.1,2,90);theta=rng.normal(size=33)
    error=check_grad(lambda v:r.loss_gradient(v,x,y,weights,.01)[0],lambda v:r.loss_gradient(v,x,y,weights,.01)[1],theta)
    assert error<2e-6


def test_readout_batch_scalar_bitwise_and_saved_model(tmp_path):
    rng=np.random.default_rng(2);x=rng.normal(size=(1000,32));model=r.StateReadout(rng.normal(size=33),SETTINGS,temporal_state.implementation_hash(SETTINGS),r.source_hash(),{})
    p=model.predict(x);np.testing.assert_array_equal(p,np.array([model.predict_one(row) for row in x],dtype=np.float32))
    path=tmp_path/'model.joblib';model.save(path);np.testing.assert_array_equal(p,r.StateReadout.load(path).predict(x))


def test_constant_feature_negative_control_fits_only_constant_score():
    class Guard:
        def partition(self,train,valid):assert not np.intersect1d(train,valid).size
    x=np.zeros((80,32));meta={'dataset_id':np.repeat(np.arange(8),10),'time_online':np.tile(np.arange(10),8),
        'target':np.tile(np.r_[np.zeros(5),np.ones(5)],8)}
    # Both classes must exist at the same time for W3 to have positive pair mass.
    flipped=meta['dataset_id']%2==1
    meta['target'][flipped]=1-meta['target'][flipped]
    hyp={'feature_settings':SETTINGS,'feature_sha256':temporal_state.implementation_hash(SETTINGS),
        'parameters':{'learner':'fixed_state_logistic','weighting':'W3','ridge':.01,'maxiter':200,'frozen_BL80_weight':.9}}
    model,p,detail=r.fit_base(x,meta,np.arange(60),np.arange(60,80),guard=Guard(),registration=hyp)
    assert np.ptp(p)==0 and np.isfinite(p).all() and detail['readout_coefficients_fitted']
    from src.scoring.ts_auc import ts_auc
    assert ts_auc(meta['target'][60:],p,meta['time_online'][60:])==.5
    with pytest.raises(ValueError):r.fit_base(x,meta,np.arange(60),np.arange(50,80),guard=Guard(),registration=hyp)
