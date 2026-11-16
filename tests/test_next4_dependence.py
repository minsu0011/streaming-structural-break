"""Causal-prefix coverage and an independent moment update oracle."""
import numpy as np
import pytest
from src.next4 import dependence as d


def settings(kind):return {'version':'joint_corr_departure_v1','representation':kind}


@pytest.mark.parametrize('kind',d.KINDS)
def test_one_hundred_random_prefix_cases_per_representation(kind):
    rng=np.random.default_rng(20260908)
    for case in range(100):
        h=rng.standard_t(5,size=128).astype(np.float32)
        a=rng.normal(size=65).astype(np.float32);b=a.copy();prefix=1+case%48
        b[prefix:]=rng.normal(0,20,len(b)-prefix)
        left=d.make_state(h,settings(kind)).replay(a)
        right=d.make_state(h,settings(kind)).replay(b)
        np.testing.assert_array_equal(left[:prefix],right[:prefix])
        state=d.make_state(h,settings(kind));stream=np.array([state.update(v).copy() for v in a],dtype=np.float32)
        np.testing.assert_array_equal(left,stream)
        assert left.shape==(65,18) and np.isfinite(left).all()


@pytest.mark.parametrize('kind',d.KINDS)
def test_constant_history_and_online_are_finite_with_fixed_state(kind):
    state=d.make_state(np.ones(128,dtype=np.float32),settings(kind))
    def arrays(value):
        if isinstance(value,np.ndarray):return [(id(value),value.shape,value.nbytes)]
        if isinstance(value,tuple):return sum((arrays(v) for v in value),[])
        return []
    before=arrays(state.args);actual=state.replay(np.ones(1024,dtype=np.float32))
    assert np.isfinite(actual).all() and arrays(state.args)==before


def test_moment_correlations_against_independent_numpy_oracle():
    rng=np.random.default_rng(471);h=rng.normal(size=256).astype(np.float32)
    state=d.make_state(h,settings('innovation'));args=state.args
    moments=args[7].copy();ring=args[5].copy();counter=int(args[6][1]);alpha=args[10]
    # Separately duplicate only the existing, immutable input normalization.
    from src.next.whitening import innovation_update
    from src.next2.complement_bank import normalized_one
    filter_args=tuple(x.copy() for x in args[0]);conditional=args[2].copy()
    for point in rng.normal(size=20):
        z=normalized_one(innovation_update(point,*filter_args),args[1],conditional)
        x=float(np.clip((z-args[3][0])/args[3][1],-8,8))
        for k,a in enumerate(alpha):
            for j,lag in enumerate(d.LAGS):
                y=ring[(counter-lag)%8]
                moments[k,j]=(1-a)*moments[k,j]+a*np.array([x,y,x*x,y*y,x*y])
        state.update(point)
        np.testing.assert_allclose(args[7],moments,rtol=0,atol=1e-14)
        for k in range(2):
            for j in range(4):
                m=moments[k,j];expected=np.clip((m[4]-m[0]*m[1])/np.sqrt(max(m[2]-m[0]**2,1e-8)*max(m[3]-m[1]**2,1e-8)),-1,1)
                assert d.correlation(m)==pytest.approx(expected,abs=1e-14)
        ring[counter]=x;counter=(counter+1)%8


@pytest.mark.parametrize('invalid',[{}, {'version':'other','representation':'innovation'},
    {'version':'joint_corr_departure_v1','representation':'innovation','online_length':100}])
def test_no_unregistered_configuration_or_future_length(invalid):
    with pytest.raises(ValueError):d.make_state(np.ones(128),invalid)
