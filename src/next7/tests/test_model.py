import pickle
import numpy as np
import pytest
from src.next7.model import fit_correction,Correction,implementation_hash
from src.next7.features import from_state


def data():
    rng=np.random.default_rng(20260913)
    x=rng.normal(size=(4096,10))
    y=(x[:,1]+rng.normal(size=4096)>.2).astype(np.uint8)
    return x,np.full(4096,.4),y,np.ones(4096)


def proof():
    return {'strict_nested_OOF':True,'outer_validation_excluded':True,'synthetic_test_only':True}


def test_rejects_in_sample_meta_provenance():
    with pytest.raises(RuntimeError,match='LEAKAGE_FAIL'):
        fit_correction('M_FULL',*data(),{'strict_nested_OOF':False})


def test_zero_weight_rows_do_not_change_the_fitted_objective():
    x,b,y,w=data()
    w[:1000]=0
    altered=y.copy()
    altered[:1000]=1-altered[:1000]
    first=fit_correction('M_FULL',x,b,y,w,proof())
    second=fit_correction('M_FULL',x,b,altered,w,proof())
    np.testing.assert_array_equal(first.beta,second.beta)
    invalid=w.copy()
    invalid[0]=-1
    with pytest.raises(ValueError):
        fit_correction('M_FULL',x,b,y,invalid,proof())


def test_offset_logistic_scalar_batch_and_serialization():
    x,b,y,w=data()
    model=fit_correction('M_FULL',x,b,y,w,proof())
    p=model.predict(b,x)
    scalar=np.array([model.predict_one(b[i],x[i]) for i in range(100)])
    np.testing.assert_allclose(p[:100],scalar,atol=1e-14,rtol=0)
    np.testing.assert_array_equal(p,pickle.loads(pickle.dumps(model)).predict(b,x))
    assert model.beta[1]>0
    assert model.training['optimizer_success']


def test_zero_correction_preserves_baseline():
    x,b,y,w=data()
    model=Correction('M_FULL',tuple(range(10)),np.zeros(10),np.ones(10),np.zeros(10),None,{},implementation_hash())
    b=np.linspace(.001,.999,len(x))
    np.testing.assert_allclose(model.predict(b,x),b,atol=1e-15,rtol=0)


def test_tree_residual_export_has_no_sigmoid_link():
    params={'n_estimators':24,'num_leaves':4,'max_depth':2,'min_child_samples':2000,
        'reg_lambda':20.,'learning_rate':.05,'random_state':20260908,'n_jobs':4,
        'deterministic':True,'force_col_wise':True,'verbosity':-1}
    x,b,y,w=data()
    m=fit_correction('I4_EVENT_SCALE_TREE',x,b,y,w,proof(),params)
    assert m.training['native_compact_max_error']<1e-12
    np.testing.assert_allclose(m.predict(b[:50],x[:50]),[m.predict_one(b[i],x[i]) for i in range(50)],atol=1e-14,rtol=0)
    assert np.max(np.abs(m.predict(b,x)-b))<.5


def test_feature_ages_bounded_and_no_forensic_time_input():
    values=np.zeros((2,30))
    values[:,5:7]=[[-1,-1],[1e12,1e12]]
    first=from_state(values)
    values[:,19:25]=1e30
    np.testing.assert_array_equal(first,from_state(values))
    assert (first[0,3:5]==0).all()
    assert (first[1,3:5]==np.log1p(16384)).all()
