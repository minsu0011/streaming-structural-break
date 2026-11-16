import json
import numpy as np
import pytest
from src.next3.orthogonal import fit_residual,predict_residual,residual_values


def sample():
    rng=np.random.default_rng(142)
    x=rng.uniform(.1,.9,(600,6)).astype(np.float32)
    x[:,1]=.1+.7*x[:,0]+.03*x[:,2]+rng.normal(0,.01,len(x))
    meta={'target':np.repeat(np.tile([0,1],15),20).astype(np.uint8),'time_online':np.tile(np.arange(20),30),
        'dataset_id':np.repeat(np.arange(30),20)}
    return x,meta


def test_weighted_fit_is_low_dimensional_and_serialization_exact():
    x,m=sample();model=fit_residual(x,m)
    assert len(model['coefficients'])==3 and model['normal_equation_max_abs_error']<1e-12
    assert np.std(residual_values(model,x))<.02
    np.testing.assert_array_equal(predict_residual(model,x),predict_residual(json.loads(json.dumps(model)),x))


def test_validation_labels_absent_and_permutation_prefix_causal():
    x,m=sample();model=fit_residual(x[:400],{k:v[:400] for k,v in m.items()})
    p=predict_residual(model,x[400:]);permutation=np.arange(200)[::-1]
    np.testing.assert_array_equal(p[permutation],predict_residual(model,x[400:][permutation]))
    np.testing.assert_array_equal(p[:7],predict_residual(model,x[400:407]))
    changed=x[400:].copy();changed[:,3:]=999.
    np.testing.assert_array_equal(p,predict_residual(model,changed))


def test_constant_inputs_and_invalid_recipe():
    x,m=sample();x[:,0]=.4;x[:,2]=1.;model=fit_residual(x,m)
    assert np.isfinite(predict_residual(model,x)).all()
    model['residual_weight']=.21
    with pytest.raises(ValueError,match='recipe'):predict_residual(model,x)
