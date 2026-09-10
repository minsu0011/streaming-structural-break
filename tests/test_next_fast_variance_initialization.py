import numpy as np
import pytest
from src.next.fast_variance_initialization import FastVarianceState,FastARCHFeatureState,fast_variance_parameters,OLD_CONFIG,OLD_POLICY
from src.next.variance_evidence import VarianceState,VarianceConfig
from src.next.normalization import fit_normalization
from src.features.arch_input import ARCHCalibratedState


@pytest.mark.parametrize('mode',['arch1','arch2','ewma'])
@pytest.mark.parametrize('n',[64,1200,2600])
def test_fast_variance_parameters_and_state_are_bitwise_exact(mode,n):
    rng = np.random.default_rng(527)
    h = rng.standard_t(5,n).astype(np.float32)
    parameters,conditional = fast_variance_parameters(h,mode)
    original = fit_normalization(h,mode)
    np.testing.assert_array_equal(parameters,original['parameters'])
    np.testing.assert_array_equal(conditional,original['conditional'])
    fast = FastVarianceState(h,VarianceConfig(normalization=mode))
    old = VarianceState(h,VarianceConfig(normalization=mode))
    np.testing.assert_array_equal(fast.parameters,old.parameters)
    np.testing.assert_array_equal(fast.conditional,old.conditional)
    np.testing.assert_array_equal(fast.calibration,old.calibration)
    points = rng.normal(size=700).astype(np.float32)
    np.testing.assert_array_equal(fast.replay(points),old.replay(points))


def test_shared_legacy_historical_setup_preserves_all_calibrations_and_features():
    rng = np.random.default_rng(387)
    h = rng.standard_t(5,1800).astype(np.float32)
    fast = FastARCHFeatureState(h)
    old = ARCHCalibratedState(h,config=OLD_CONFIG,policy=OLD_POLICY)
    for name in ('center','scale','mask'):
        np.testing.assert_array_equal(getattr(fast.inner,name),getattr(old.inner,name))
    points = rng.normal(size=750).astype(np.float32)
    expected = np.array([old.update_and_get(x).copy() for x in points])
    actual = np.array([fast.update_and_get(x).copy() for x in points])
    np.testing.assert_array_equal(actual,expected)
