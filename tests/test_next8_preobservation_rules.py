"""Future current-point changes cannot alter the already available rule."""
import src.next8
import copy
import numpy as np
from tests.test_next8_streaming_runtime import actual_cache,fitted_fixture
from src.next8.streaming_runtime import PreparedJoint
from src.next8.preobservation_rules import preview


def test_direction_preview_precedes_current_F_and_preserves_memory(actual_cache):
    cache,h,online=actual_cache;model,selection,graph,_=fitted_fixture(cache,'M5')
    prepared=PreparedJoint(model,selection,'M5',{0:graph},resident=False)
    a=prepared.new_state(h);a.replay(online[:31]);b=a.snapshot();counter=a.counter
    before=preview(a);again=preview(a)
    assert before==again and a.counter==counter and before['time_online']==31
    a.predict_one(online[31]);b.predict_one(online[31]+20)
    for k,rule in enumerate(before['rules']):
        assert a.own_orientation_condition[k]==b.own_orientation_condition[k]==rule['own_or_H_condition']
        assert a.forecast_graph_condition[k]==b.forecast_graph_condition[k]==rule['forecast_graph_condition']
        assert not rule['current_F_observed'] and not rule['future_state_confirmation_observed']
    unpeeked=prepared.new_state(h);unpeeked.replay(online[:31]);unpeeked.predict_one(online[31])
    np.testing.assert_array_equal(a.replay(online[32:]),unpeeked.replay(online[32:]))
