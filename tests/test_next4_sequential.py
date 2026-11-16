import numpy as np
import pytest
from scipy.special import logsumexp
from src.next4 import sequential as d


def settings(kind):return {'version':'discrete_likelihood_v1','kind':kind}


@pytest.mark.parametrize('kind',d.KINDS)
def test_one_hundred_random_prefix_cases(kind):
    rng=np.random.default_rng(20260908)
    for case in range(100):
        h=rng.standard_t(5,size=128).astype(np.float32);a=rng.normal(size=65).astype(np.float32)
        prefix=1+case%48;b=a.copy();b[prefix:]=rng.normal(0,20,len(b)-prefix)
        actual=d.make_state(h,settings(kind)).replay(a)
        np.testing.assert_array_equal(actual[:prefix],d.make_state(h,settings(kind)).replay(b)[:prefix])
        state=d.make_state(h,settings(kind));expected=np.array([state.update(v).copy() for v in a])
        np.testing.assert_array_equal(actual,expected)
        assert actual.shape==(65,len(d.feature_names(settings(kind)))) and np.isfinite(actual).all()


@pytest.mark.parametrize('kind',d.KINDS)
def test_long_degenerate_input_fixed_finite_state(kind):
    state=d.make_state(np.ones(128),settings(kind))
    arrays=[v for v in state.args if isinstance(v,np.ndarray)];identity=[(id(v),v.shape,v.nbytes) for v in arrays]
    actual=state.replay(np.zeros(16384))
    assert np.isfinite(actual).all()
    assert identity==[(id(v),v.shape,v.nbytes) for v in arrays]


@pytest.mark.parametrize('kind',['mean_mixture','scale_mixture'])
def test_sparse_windows_against_direct_slices_and_scipy(kind):
    from src.next.whitening import innovation_update
    from src.next2.complement_bank import normalized_one
    rng=np.random.default_rng(153);state=d.make_state(rng.normal(size=256),settings(kind));args=state.args
    filt=tuple(v.copy() for v in args[0]);conditional=args[2].copy();zs=[]
    for point in rng.normal(size=1100):
        z=normalized_one(innovation_update(point,*filt),args[1],conditional)
        z=float(np.clip((z-args[3][0])/args[3][1],-6,6));zs.append(z);actual=state.update(point).copy()
        expected=[];totals=[]
        for age in d.AGES:
            values=np.asarray(zs[-age:]);n=len(values)
            if kind=='mean_mixture':
                ll=d.MEANS*values.sum()-.5*n*d.MEANS**2;lo=ll[:3];hi=ll[3:]
            else:
                ll=-.5*n*np.log(d.VARIANCES)+.5*(1-1/d.VARIANCES)*np.sum(values**2);lo=ll[:2];hi=ll[2:]
            total=logsumexp(ll)-np.log(len(ll));lower=logsumexp(lo)-np.log(len(lo));upper=logsumexp(hi)-np.log(len(hi))
            expected.extend([total,lower,upper,upper-lower]);totals.append(total)
        expected.extend([logsumexp(totals)-np.log(8),max(totals),logsumexp(totals[:4])-np.log(4),logsumexp(totals[4:])-np.log(4)])
        np.testing.assert_allclose(actual,np.array(expected,dtype=np.float32),rtol=2e-6,atol=2e-6)


def test_hazard_posterior_against_probability_space_filter():
    rng=np.random.default_rng(17);states=np.full((3,11),-np.inf);states[:,0]=0.;prob=np.zeros((3,11));prob[:,0]=1.
    output=np.empty(20)
    for z in rng.normal(size=100):
        ratios=np.exp(np.r_[d.MEANS*z-.5*d.MEANS**2,-.5*np.log(d.VARIANCES)+.5*(1-1/d.VARIANCES)*z*z])
        for j,h in enumerate(d.HAZARDS):
            prior_null=prob[j,0];prob[j,1:]=(prob[j,1:]+prior_null*h/10)*ratios
            prob[j,0]=prior_null*(1-h);prob[j]/=prob[j].sum()
        d.hazard_features(z,states,output)
        np.testing.assert_allclose(np.exp(states),prob,atol=1e-13,rtol=1e-12)
        assert np.isfinite(output).all()


@pytest.mark.parametrize('invalid',[{}, {'version':'discrete_likelihood_v1','kind':'mean_mixture','final_online_length':512}])
def test_reject_unregistered_settings(invalid):
    with pytest.raises(ValueError):d.make_state(np.ones(128),invalid)
