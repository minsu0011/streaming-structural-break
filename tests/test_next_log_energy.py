import numpy as np
import pytest
from src.next.log_energy_evidence import LogEnergyConfig,LogEnergyState,energy_channels
from src.next.whitening import fit_historical,innovation_update


@pytest.mark.parametrize('alignment,cap',[('current',False),('lagged',False),('current',True)])
def test_channels_match_direct_energy_and_prefix_after_ring_wrap(alignment,cap):
    rng = np.random.default_rng(9208)
    h = rng.standard_t(5,1700).astype(np.float32)
    o = rng.normal(size=731).astype(np.float32)
    o[310] = 100.
    config = LogEnergyConfig(alignment=alignment,cap_at_h99=cap)
    state = LogEnergyState(h,config)
    fit = fit_historical(h,8)
    args = tuple(fit[k] for k in ('reference','coefficients','ring','counter'))
    previous = state.previous[0]
    expected = []
    params,calibration = state.parameters.copy(),state.calibration.copy()
    for point in o:
        residual = innovation_update(float(point),*args)
        square = (residual-params[0])**2
        current = np.log1p((min(square,params[2]) if cap else square)/params[1])
        channels = np.array([square/params[1]-1.,previous if alignment=='lagged' else current])
        expected.append(np.clip((channels-calibration[0])/calibration[1],-12.,12.))
        previous = current
    full = state.replay(o)
    np.testing.assert_allclose(full[:,[0,13]],np.asarray(expected).astype(np.float32),rtol=0,atol=0)
    for end in (1,129,513,731):
        prefix = LogEnergyState(h,config).replay(o[:end])
        np.testing.assert_array_equal(prefix,full[:end])
    np.testing.assert_array_equal(LogEnergyState(h,config).replay(o),full)
    initial_size = LogEnergyState(h,config).state_array_bytes
    assert state.state_array_bytes==initial_size


def test_cap_is_historical_and_lagged_channel_uses_previous_energy():
    parameters = np.array([0.,1.,4.])
    previous = np.array([.3])
    output = np.empty(2)
    energy_channels(100.,parameters,previous,True,True,output)
    assert output[0]==9999.
    assert output[1]==.3
    assert previous[0]==np.log(5.)
    energy_channels(0.,parameters,previous,True,True,output)
    assert output[1]==np.log(5.)
    np.testing.assert_array_equal(parameters,[0.,1.,4.])
