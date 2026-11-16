import json
import numpy as np
import pytest
from src.next3.learned_router import fit_gate
from src.next3.gate_serialization import describe_gate,descriptor_predict


@pytest.mark.parametrize('kind',['R2_LOGISTIC_GATE','R3_DEPTH2_GATE'])
def test_json_gate_replays_native_predictions_exactly(kind):
    rng=np.random.default_rng(291)
    train=rng.uniform(.01,.99,(2000,6)).astype(np.float32)
    md={'target':rng.integers(0,2,2000),'time_online':np.arange(2000)%40,'dataset_id':np.arange(2000)//40}
    gate=fit_gate(kind,train,md)
    descriptor=json.loads(json.dumps(describe_gate(gate),sort_keys=True,allow_nan=False))
    test=rng.uniform(.01,.99,(1000,6)).astype(np.float32)
    np.testing.assert_array_equal(descriptor_predict(descriptor,test),gate.predict(test))
