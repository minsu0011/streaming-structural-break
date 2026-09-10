"""Real serialized candidates: cached features, causal streams and blend rounding."""
from pathlib import Path
import json
import numpy as np
import pytest
from src.next2.io import model_class
from src.next2.blend import FixedBlendBundle
from src.next2.runtime import prepare

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('eid',['C205_DEPENDENCE','M2_DECAYED_ONLY','J201_JOINT_DEPENDENCE_EWMA32'])
def test_prepared_next2_bank_exact_and_prefix_causal(eid):
    folder=ROOT/'artifacts/next2/experiments'/eid
    if not (folder/'RESULTS.json').exists():
        pytest.skip('Research model not present in this source-only checkout')
    candidate=json.loads((folder/'RESULTS.json').read_text())['candidate']
    model=model_class(candidate).load(folder/'full/fold_0.joblib')
    rng=np.random.default_rng(4817)
    h=rng.standard_t(5,640).astype(np.float32)
    o=rng.normal(size=259).astype(np.float32)
    original=model.make_state(h)
    expected=np.asarray([model.predict_one(original.update(p)) for p in o],dtype=np.float32)
    engine=prepare(model)
    live=engine.new_state(h)
    actual=np.asarray([live.predict_one(p) for p in o],dtype=np.float32)
    np.testing.assert_array_equal(actual,expected)
    prefix=engine.new_state(h)
    np.testing.assert_array_equal([prefix.predict_one(p) for p in o[:91]],actual[:91])
    assert live.state_array_bytes>0 and live.immutable_tree_array_bytes>0


def test_prepared_blend_equals_frozen_component_rounding():
    path=ROOT/'artifacts/next2/experiments/BL2_C205_DEPENDENCE_80/full/fold_0.joblib'
    if not path.exists():
        pytest.skip('Research model not present in this source-only checkout')
    model=FixedBlendBundle.load(path)
    rng=np.random.default_rng(88172)
    h=rng.normal(size=512).astype(np.float32)
    o=rng.normal(size=145).astype(np.float32)
    original=model.make_state(h)
    expected=np.asarray([model.predict_one(original.update(p)) for p in o],dtype=np.float32)
    state=prepare(model).new_state(h)
    actual=np.asarray([state.predict_one(p) for p in o],dtype=np.float32)
    np.testing.assert_array_equal(actual,expected)
    assert state.state_array_bytes==state.primary.state_array_bytes+state.complement.state_array_bytes
