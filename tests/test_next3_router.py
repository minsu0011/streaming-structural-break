import numpy as np
import pytest
from src.next3.router import heuristic_predict,heuristic_weight,blend_with_weight


def test_score_gap_fixed_boundaries_and_low_confidence_fallback():
    features=np.array([[.5,.2,0,0,0,0],[.5,.45,0,0,0,0],[.5,.5,0,0,0,0],[.5,.55,0,0,0,0],[.5,.8,0,0,0,0]],dtype=np.float32)
    pred,w=heuristic_predict(features,'R1_SCORE_GAP')
    np.testing.assert_array_equal(w,[0,.1,.1,.2,.3])
    assert pred[0]==features[0,0]
    assert pred.dtype==np.float32
    assert np.all((pred>=np.minimum(features[:,0],features[:,1]))&(pred<=np.maximum(features[:,0],features[:,1])))


def test_ratio_routing_is_bounded_and_scale_suppresses_specialist():
    features=np.array([[.5,.7,0,.5,0,0],[.5,.7,0,1,0,0],[.5,.7,0,1.5,0,0],[.5,.7,0,3,0,0],[.5,.7,0,5,0,0],[.5,.7,100,5,0,0]],dtype=np.float32)
    np.testing.assert_array_equal(heuristic_weight(features,'R1_EVIDENCE_RATIO'),[0,0,.1,.2,.3,0])


def test_router_row_permutation_and_prefix_do_not_change_predictions():
    rng=np.random.default_rng(73)
    features=rng.uniform(0,1,(500,6)).astype(np.float32)
    features[:,2:4]*=8
    for kind in ('R1_SCORE_GAP','R1_EVIDENCE_RATIO'):
        pred,_=heuristic_predict(features,kind)
        prefix,_=heuristic_predict(features[:100],kind)
        np.testing.assert_array_equal(pred[:100],prefix)
        changed=features.copy();changed[100:]=999
        other,_=heuristic_predict(changed[:100],kind)
        np.testing.assert_array_equal(other,prefix)
        order=rng.permutation(500)
        reordered,_=heuristic_predict(features[order],kind)
        np.testing.assert_array_equal(reordered,pred[order])


@pytest.mark.parametrize('weight',[np.array([-.1]),np.array([.31]),np.array([np.nan])])
def test_unregistered_or_nonfinite_weight_fails(weight):
    with pytest.raises(ValueError):blend_with_weight(np.array([.5]),np.array([.7]),weight)
