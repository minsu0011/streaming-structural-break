from types import SimpleNamespace
import numpy as np
import pytest
from src.next.variance_evidence import VarianceState,VarianceConfig
from src.next.pruned_variance_runtime import pruned_variance_step,supports_pruned_pair


@pytest.mark.parametrize('mode',['arch1','arch2','ewma'])
def test_omitting_four_channels_preserves_all_26_consumed_features(mode):
    rng = np.random.default_rng(9274)
    h = rng.standard_t(5,1600).astype(np.float32)
    points = rng.normal(size=1800).astype(np.float32)
    points[800:] *= 2
    config = VarianceConfig(normalization=mode)
    expected = VarianceState(h,config).replay(points)[:,:26]
    state = VarianceState(h,config)
    actual = []
    for point in points:
        output = pruned_variance_step(float(point),state.filter_args,state.parameters,state.conditional,state.mode,state.calibration,state.work,state.evidence_args)
        actual.append(output[:26].copy())
    np.testing.assert_array_equal(np.asarray(actual),expected)
    # Unused state remains untouched rather than collecting online history.
    assert np.count_nonzero(state.evidence_args[2][2:])==0


def test_pruning_refuses_a_model_that_consumes_an_omitted_channel():
    model = SimpleNamespace(extension={'kind':'conditional_variance'},names=('arch8__current_z','conditional_variance__raw_energy_glr_max'))
    assert supports_pruned_pair(model)
    model.names += ('conditional_variance__variance_lag1_score_current',)
    assert not supports_pruned_pair(model)
