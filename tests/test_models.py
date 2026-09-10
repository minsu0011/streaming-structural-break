import numpy as np
import pytest
from scipy.special import expit
from threadpoolctl import threadpool_limits
from src.features.streaming import replay
from src.validation.synthetic import fixture
from src.data.labels import labels_from_tau
from src.models.learners import MODEL_NAMES,fit_candidate,ModelBundle

@pytest.fixture(scope='module')
def engineering_data():
    xs=[]; ys=[]
    for j,kind in enumerate(['no_break','mean_up','mean_down','variance_up','variance_down','ar_up']):
        h,o,tau=fixture(kind,seed=j)
        xs.extend(replay(h,o)); ys.extend(labels_from_tau(len(o),tau))
    return np.asarray(xs),np.asarray(ys)

@pytest.mark.parametrize('name',MODEL_NAMES)
def test_save_load_and_single_row_parity(name,engineering_data,tmp_path):
    x,y=engineering_data
    with threadpool_limits(limits=1):
        model=fit_candidate(name,x,y)
        expected=model.predict(x[:37])
        model.save(tmp_path/'model.joblib')
        restored=ModelBundle.load(tmp_path/'model.joblib')
        np.testing.assert_array_equal(expected,restored.predict(x[:37]))
        live=[restored.predict_one(row) for row in x[:37]]
    np.testing.assert_allclose(expected,live,rtol=0,atol=1e-12)
    assert np.isfinite(live).all() and min(live)>=0 and max(live)<=1
    if model.linear_weight is not None:
        transformed=model.scaler.transform(x[:37,model.columns])
        official=expit(model.estimator.decision_function(transformed))
        np.testing.assert_allclose(expected,official,rtol=0,atol=2e-6)
    elif model.compact is not None:
        probe=np.r_[x[:200,model.columns],np.random.default_rng(42).normal(size=(200,len(model.columns))).astype(np.float32)*10]
        with threadpool_limits(limits=1):
            native=model.estimator.booster_.predict(probe,num_threads=1) if name=='M4_lightgbm' else model.estimator.predict_proba(probe)[:,1]
        np.testing.assert_allclose(model.compact.predict(probe),native,rtol=0,atol=1e-12)

def test_model_hash_guard(engineering_data,tmp_path):
    x,y=engineering_data; model=fit_candidate('M1_logistic',x,y)
    model.feature_hash='invalid'; model.save(tmp_path/'model.joblib')
    with pytest.raises(RuntimeError): ModelBundle.load(tmp_path/'model.joblib')
