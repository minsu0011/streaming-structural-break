"""Shared update cannot alter any head or saved causal state."""
import src.next8
import copy
import numpy as np
import pytest
import joblib
from tests.test_next8_streaming_runtime import actual_cache,fitted_fixture
from src.next8.streaming_runtime import PreparedJoint
from src.next8.multihead_runtime import PreparedMultiHead


def test_multihead_matches_separate_engines_and_saved_prefix(actual_cache,tmp_path):
    cache,h,online=actual_cache;heads={}
    for method in ('M1','M2','M3','M4','M5'):
        model,selection,graph,_=fitted_fixture(cache,method)
        heads[method]=PreparedJoint(model,selection,method,{0:graph},resident=False)
    prepared=PreparedMultiHead(heads);state=prepared.new_state(h)
    prefix=state.replay(online[:57]);saved=state.snapshot();joblib.dump(saved,tmp_path/'multihead.joblib')
    suffix=state.replay(online[57:]);restored=joblib.load(tmp_path/'multihead.joblib')
    np.testing.assert_array_equal(suffix,restored.replay(online[57:]))
    result=np.vstack([prefix,suffix])
    for k,item in enumerate(heads.values()):
        np.testing.assert_array_equal(result[:,k],item.new_state(h).replay(online))
    assert state.counter==len(online) and saved.counter==57
    changed=online.copy();changed[57:]*=-7
    np.testing.assert_array_equal(prepared.new_state(h).replay(changed)[:57],prefix)
    scalar=prepared.new_state(h)
    np.testing.assert_array_equal(np.asarray([scalar.predict_one(x) for x in online[:57]]),prefix)


def test_multihead_rejects_different_selection_or_graph(actual_cache):
    cache,_,_=actual_cache;model,selection,graph,_=fitted_fixture(cache,'M5')
    a=PreparedJoint(model,selection,'M5',{0:graph},resident=False);b=copy.deepcopy(a)
    b.graph_models[0].own_table[0,0]+=.001
    with pytest.raises(ValueError,match='same selection'):PreparedMultiHead({'a':a,'b':b})
    b=copy.deepcopy(a);b.selection['variable_features']=[]
    with pytest.raises(ValueError,match='same selection'):PreparedMultiHead({'a':a,'b':b})
