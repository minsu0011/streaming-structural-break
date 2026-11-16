import numpy as np
import pytest
from scipy.spatial.distance import jensenshannon
from src.next4 import distribution as d


def settings(kind):return {'version':'fixed_residual_cells_v1','kind':kind}


@pytest.mark.parametrize('kind',d.KINDS)
def test_one_hundred_prefix_streaming_cases(kind):
    rng=np.random.default_rng(20260908)
    for case in range(100):
        h=rng.standard_t(5,size=128).astype(np.float32);a=rng.normal(size=65).astype(np.float32)
        prefix=1+case%48;b=a.copy();b[prefix:]=rng.normal(0,20,len(b)-prefix)
        actual=d.make_state(h,settings(kind)).replay(a)
        np.testing.assert_array_equal(actual[:prefix],d.make_state(h,settings(kind)).replay(b)[:prefix])
        state=d.make_state(h,settings(kind));expected=np.array([state.update(v).copy() for v in a])
        np.testing.assert_array_equal(actual,expected)
        assert actual.shape==(65,26) and np.isfinite(actual).all()


@pytest.mark.parametrize('kind',d.KINDS)
def test_degenerate_stream_bounded_state_reference_immutable(kind):
    state=d.make_state(np.ones(128,dtype=np.float32),settings(kind));reference=state.args[3].copy()
    masses=state.args[4];identity=(id(masses),masses.shape,masses.nbytes)
    output=state.replay(np.ones(4096,dtype=np.float32))
    assert np.isfinite(output).all()
    assert identity==(id(state.args[4]),state.args[4].shape,state.args[4].nbytes)
    np.testing.assert_array_equal(reference,state.args[3])
    np.testing.assert_allclose(masses.sum(axis=1),1.,atol=2e-14)


def test_divergences_against_scipy_and_independent_hellinger():
    rng=np.random.default_rng(735)
    for _ in range(100):
        p=rng.dirichlet(np.ones(8));q=rng.dirichlet(np.ones(8))
        js=d.summaries(p,q,1);he=d.summaries(p,q,0)
        assert js[0]==pytest.approx(jensenshannon(p,q)**2,abs=1e-14)
        assert he[0]==pytest.approx(.5*np.sum((np.sqrt(p)-np.sqrt(q))**2),abs=1e-14)
        for out in (js,he):
            assert out[0]==pytest.approx(out[1]+out[2],abs=1e-14)
            assert out[0]==pytest.approx(out[3]+out[4]+out[5],abs=1e-14)
            assert out[0]>=0
        np.testing.assert_allclose(d.summaries(p,p,2),0.,atol=1e-14)


def test_histogram_recurrence_independent_numpy_oracle():
    from src.next.whitening import innovation_update
    from src.next2.complement_bank import normalized_one
    rng=np.random.default_rng(128);state=d.make_state(rng.normal(size=256),settings('hellinger'))
    args=state.args;filt=tuple(v.copy() for v in args[0]);conditional=args[2].copy();mass=args[4].copy()
    for v in rng.normal(size=100):
        z=normalized_one(innovation_update(v,*filt),args[1],conditional)
        one=np.eye(8)[np.searchsorted(d.EDGES,z,side='right')]
        mass=(1-args[5][:,None])*mass+args[5][:,None]*one
        state.update(v)
        np.testing.assert_allclose(state.args[4],mass,atol=1e-14,rtol=0)


@pytest.mark.parametrize('invalid',[{}, {'version':'fixed_residual_cells_v1','kind':'js','tau':100}])
def test_reject_unregistered_settings(invalid):
    with pytest.raises(ValueError):d.make_state(np.ones(128),invalid)
