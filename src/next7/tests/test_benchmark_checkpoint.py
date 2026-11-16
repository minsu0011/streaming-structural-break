import pickle
import numpy as np
import pytest
from src.next7.benchmark_adapter import Factory, CLASSIFIERS, memory, checkpoint, restore
from src.next7.common import ROOT, read_json
from src.next6_extension.synthetic import generated


@pytest.mark.parametrize('method', CLASSIFIERS)
def test_real_scalar_state_checkpoint_restores_exact_continuation(method):
    h, o, _, _ = generated('RETURN_NORMAL', 0, partition=31, horizon=1024, control=False,
        lock=read_json(ROOT / 'COUNTEREXAMPLE_SUITE_LOCK.json'))
    factory = Factory(method)
    state = factory.new_state(h)
    for value in o[:257]:
        state.predict_one(float(value))
    before = memory(method, state)
    payload = pickle.loads(pickle.dumps(checkpoint(method, state), protocol=5))
    resumed = restore(factory, payload)
    assert memory(method, resumed) == before
    left = np.array([state.predict_one(float(v)) for v in o[257:]])
    right = np.array([resumed.predict_one(float(v)) for v in o[257:]])
    np.testing.assert_array_equal(left, right)
    assert memory(method, state) == before and memory(method, resumed) == before
