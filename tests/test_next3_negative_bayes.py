import numpy as np
from src.next3.negative_bayes import NegativeBayesState,NAMES,negative_feature_replay
from src.next3.frozen import candidate,J102
from src.next.candidate_io import model_class
from src.next3.artifacts import ROOT


def test_negative_branch_keeps_all_frozen_features_and_reuses_state():
    rng=np.random.default_rng(145)
    h=rng.normal(size=512).astype(np.float32);o=rng.normal(size=180).astype(np.float32)
    spec,folder=candidate(J102)
    model=model_class(spec).load(folder/'full/fold_0.joblib')
    old=model.make_state(h)
    state=NegativeBayesState(h,tuple(model.names)+NAMES,spec['extension'],spec['extension_registration'])
    actual=[]
    for point in o:
        before=old.update(point).copy();after=state.update(point).copy()
        np.testing.assert_array_equal(before,after[:-4]);actual.append(after[-4:])
    np.testing.assert_array_equal(np.asarray(actual),negative_feature_replay(h,o))
    np.testing.assert_array_equal(negative_feature_replay(h,o[:75]),np.asarray(actual)[:75])
