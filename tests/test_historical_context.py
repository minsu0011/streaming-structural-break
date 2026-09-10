import numpy as np
import pytest
from src.features.config import CONFIG
from src.features.historical_calibration import CalibrationPolicy
from src.features.ar_residual_input import ARResidualCalibratedState
from src.features.historical_context import HistoricalContextState,names,model_families


@pytest.mark.parametrize('historical',[[],[1.],[2.]*300,[np.nan,np.inf,0.,-1.]*100])
def test_historical_context_is_finite_fixed_and_preserves_abcd(historical):
    config=CONFIG.variant(normalization='median_mad',scales=(5,20,160));policy=CalibrationPolicy(method='historical_median_mad')
    original=ARResidualCalibratedState(historical,config=config,policy=policy);context=HistoricalContextState(historical,config=config,policy=policy)
    before=context.state_array_bytes;fixed=context.context.copy()
    for point in [0.,1.,np.nan,-2.,50.]*30:
        left=original.update_and_get(point).copy();right=context.update_and_get(point).copy()
        np.testing.assert_array_equal(left[:51],right[:51]);np.testing.assert_array_equal(right[51:],fixed)
        assert np.isfinite(right).all()
    assert context.state_array_bytes==before and len(names(config))==57
    assert model_families('ABCDG')=='ABCDEFQ' and model_families('G')=='EFQ'
