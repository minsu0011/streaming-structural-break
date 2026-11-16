from pathlib import Path
import pickle
import numpy as np
import pytest
from src.next7.common import ROOT, BASE, read_json
from src.next7.model import Correction
from src.next7.calibration import load_calibration, cutoff
from src.next7.integrated_runtime import Prepared
from src.next7.event import CombinedState
from src.next7.null_growth import QualifiedState, qualified_features
from src.next7.features import from_state
from src.next5.model import FeatureModel
from src.next5.runtime import SharedResearchBaseline
from src.next6_extension.synthetic import generated


METHODS = ('M_EVENT', 'I3_CURRENT_EVENT', 'M_FULL', 'I4_EVENT_SCALE_TREE', 'G2_EVENT', 'G2_FULL', 'G3_PAIRWISE_FULL')


def models(fold, method, resident=False):
    refs = [FeatureModel.load(ROOT / f'artifacts/next5/references/fits/PRIMARY/full/{k}/fold_{fold}/model.joblib') for k in ('J102', 'C205')]
    if method in ('M_EVENT', 'G2_EVENT'):
        head = None
    else:
        relative = f'g3/fold_{fold}' if method == 'G3_PAIRWISE_FULL' else f'g2/fold_{fold}/G2_FULL' if method == 'G2_FULL' else f'g1/fold_{fold}/{method}'
        head = Correction.load(BASE / relative / 'model.joblib')
    return Prepared(*refs, head, load_calibration(), qualified=method.startswith(('G2_', 'G3_')), resident=resident)


def workload(case='RETURN_NORMAL', length=1024):
    return generated(case, 0, partition=31, horizon=length, control=False, lock=read_json(ROOT / 'COUNTEREXAMPLE_SUITE_LOCK.json'))[:2]


@pytest.mark.parametrize('fold', [0, 4])
@pytest.mark.parametrize('method', METHODS)
def test_fused_matches_frozen_baseline_and_readable_state_head(fold, method):
    h, o = workload()
    p = models(fold, method)
    state = p.new_state(h)
    actual = state.replay(o)
    reference = p.baseline.new_state(h)
    baseline = np.array([reference.predict_one(x) for x in o])
    component = QualifiedState if p.qualified else CombinedState
    raw = component(h, core_cut=p.calibration['core_cut'], scale_cut=cutoff(p.calibration, len(h))).replay(o)
    features = qualified_features(raw) if p.qualified else from_state(raw)
    expected = .9 * baseline + .1 * features[:, 1] if p.correction is None else p.correction.predict(baseline, features)
    np.testing.assert_allclose(actual[:, 0], baseline, rtol=0., atol=1e-12)
    np.testing.assert_allclose(actual[:, 1], expected, rtol=0., atol=1e-12)
    np.testing.assert_array_equal(state.snapshot(), raw[-1])
    np.testing.assert_allclose(state.args[-2], features[-1], rtol=0., atol=1e-14)


@pytest.mark.parametrize('method', ['I4_EVENT_SCALE_TREE', 'G2_FULL', 'G3_PAIRWISE_FULL'])
def test_scalar_prefix_interleaving_pickle_and_no_shared_mutable_head(method):
    h, o = workload()
    p = models(0, method, resident=True)
    full = p.new_state(h).replay(o)
    a, b = p.new_state(h), p.new_state(h)
    first = np.array([a.predict_one(x) for x in o[:137]])
    np.testing.assert_array_equal(first, full[:137, 1])
    snapshot = pickle.dumps(a)
    clone = pickle.loads(snapshot)
    fork = a.clone()
    for index in range(137, 257):
        assert a.predict_one(o[index]) == clone.predict_one(o[index]) == fork.predict_one(o[index]) == full[index, 1]
        b.predict_one(o[index] + 7.)
    np.testing.assert_array_equal(p.new_state(h).replay(o[:137]), full[:137])


def test_invalid_scalar_or_future_batch_input_does_not_advance_stream():
    h, o = workload()
    p = models(0, 'G2_EVENT')
    a = p.new_state(h)
    a.replay(o[:64])
    expected = a.clone().replay(o[64:80])
    for invalid in [np.nan, np.inf, -np.inf, 1e100]:
        with pytest.raises(ValueError):
            a.predict_one(invalid)
        with pytest.raises(ValueError):
            a.replay([o[64], invalid])
    np.testing.assert_array_equal(a.replay(o[64:80]), expected)


def test_short_H_abstains_only_qualified_memory_and_retains_baseline():
    h, o = workload('V4')
    h = h[-512:]
    p = models(0, 'G2_EVENT')
    state = p.new_state(h)
    actual = state.replay(o)
    assert not state.admission_calibration['reference_valid']
    assert state.snapshot()[30] == 0 and state.snapshot()[43] == 0
    np.testing.assert_array_equal(actual[:, 1], .9 * actual[:, 0])
    assert np.ptp(actual[:, 0]) > 0
