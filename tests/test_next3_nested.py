import numpy as np
import pytest
from src.next3.nested import nested_partitions
from src.next3.learned_router import logistic_objective,fit_gate


class Guard:
    def partition(self, train, valid):
        assert not set(train)&set(valid)


def test_nested_base_training_never_sees_outer_validation():
    metadata={'dataset_id':np.repeat(np.arange(15),3),'fold':np.repeat(np.arange(15)%5,3)}
    train=np.flatnonzero(metadata['fold']!=2);valid=np.flatnonzero(metadata['fold']==2)
    partitions=nested_partitions(metadata,train,valid,Guard())
    seen=[]
    for fold,inner_train,inner_valid in partitions:
        assert not set(valid)&set(inner_train)
        assert not set(valid)&set(inner_valid)
        assert not set(metadata['dataset_id'][inner_train])&set(metadata['dataset_id'][inner_valid])
        seen.extend(inner_valid)
    np.testing.assert_array_equal(np.sort(seen),train)


def test_duplicate_or_overlapping_outer_rows_are_rejected():
    metadata={'dataset_id':np.arange(20),'fold':np.arange(20)%5}
    with pytest.raises(ValueError):nested_partitions(metadata,[1,1,2],[3],Guard())
    with pytest.raises(ValueError):nested_partitions(metadata,[1,2],[2],Guard())


def test_logistic_mixture_gradient_matches_finite_difference():
    rng=np.random.default_rng(199)
    x=np.column_stack([np.ones(80),rng.normal(size=(80,6))])
    p=rng.uniform(.05,.95,80);c=rng.uniform(.05,.95,80)
    y=rng.integers(0,2,80);weight=rng.uniform(.2,2,80);beta=rng.normal(size=7)*.2
    loss,gradient=logistic_objective(beta,x,p,c,y,weight)
    for k in range(len(beta)):
        step=np.zeros(len(beta));step[k]=1e-5
        high=logistic_objective(beta+step,x,p,c,y,weight)[0]
        low=logistic_objective(beta-step,x,p,c,y,weight)[0]
        assert gradient[k]==pytest.approx((high-low)/2e-5,abs=1e-8)


@pytest.mark.parametrize('kind',['R2_LOGISTIC_GATE','R3_DEPTH2_GATE'])
def test_learned_gate_uses_only_feature_input_after_fit_and_is_bounded(kind):
    rng=np.random.default_rng(52)
    x=rng.uniform(.02,.98,(800,6)).astype(np.float32)
    metadata={'target':rng.integers(0,2,800),'time_online':np.arange(800)%20,'dataset_id':np.arange(800)//20}
    model=fit_gate(kind,x,metadata)
    prediction=model.predict(x)
    weight=model.weights(x)
    assert np.isfinite(prediction).all() and ((weight>=0)&(weight<=.3)).all()
    np.testing.assert_array_equal(model.predict(x[:45]),prediction[:45])
