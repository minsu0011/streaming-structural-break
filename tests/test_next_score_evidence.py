import numpy as np
import pytest
from src.next.score_evidence import ARScoreState,ScoreConfig,SCORE_NAMES


@pytest.mark.parametrize('order',[4,8,12])
def test_ar_score_batch_stream_and_future_invariance(order):
    rng=np.random.default_rng(7451)
    h=rng.standard_t(5,1400).astype(np.float32)
    o=rng.normal(size=300).astype(np.float32)
    state=ARScoreState(h,ScoreConfig(order=order))
    stream=np.array([state.update(v).copy() for v in o[:170]])
    batch=ARScoreState(h,ScoreConfig(order=order)).replay(o)
    np.testing.assert_array_equal(stream,batch[:170])
    assert np.isfinite(batch).all()


def test_ar_score_glr_matches_direct_vector_score_sums():
    rng=np.random.default_rng(339)
    h=rng.normal(size=1200).astype(np.float32)
    state=ARScoreState(h)
    points=rng.normal(size=290).astype(np.float32)
    whitened=[]
    for t,point in enumerate(points,1):
        output=state.update(point).copy()
        whitened.append(state.args[13][2].copy())
        scores=np.asarray(whitened)
        ages=[age for age in [1,2,4,8,16,32,64,128] if age<=t]
        expected=max(.5*np.sum(scores[-age:].sum(axis=0)**2)/age for age in ages)
        np.testing.assert_allclose(output[SCORE_NAMES.index('score_glr_max')],np.float32(expected),atol=3e-5,rtol=1e-6)


def test_ar_coefficient_change_response_without_changing_innovation_variance():
    rng=np.random.default_rng(8376)
    values=np.empty(7000,dtype=np.float32)
    values[0]=0
    for t in range(1,len(values)):
        phi=.4 if t<5500 else .75
        sigma=1. if t<5500 else np.sqrt((1-.75**2)/(1-.75**2+(.75-.4)**2))
        values[t]=phi*values[t-1]+sigma*rng.normal()
    h=values[:5000]
    features=ARScoreState(h).replay(values[5000:])
    col=SCORE_NAMES.index('score_ewma_q_128')
    assert np.median(features[1000:,col])>2*np.median(features[128:450,col])


def test_ar_score_fixed_memory_and_immutable_h_fit():
    rng=np.random.default_rng(528)
    state=ARScoreState(rng.normal(size=1000))
    size=state.state_array_bytes
    saved={j:state.args[j].copy() for j in [0,1,4,5,6]}
    state.replay(rng.normal(size=20000))
    assert size==state.state_array_bytes
    for j,value in saved.items():
        np.testing.assert_array_equal(value,state.args[j])
