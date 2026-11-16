import src.next8
import numpy as np
import pytest
from src.next8.forecast_causal_replay import bits_equal, saved_parity, fresh_variants
from src.next8.forecasting import forecast_designs
from src.next8.transitions import transition_inputs, transition_tabular_design


def test_actual_D_T_designs_use_prefix_without_future_target():
    rng = np.random.default_rng(8101642)
    h = rng.normal(size=512).astype(np.float32)
    o = rng.normal(size=50).astype(np.float32)
    variants = fresh_variants(123, h, o, prefix=33)
    rows = np.arange(0, 33, 8)
    expected = None
    for cache in variants:
        with pytest.raises(AssertionError, match='labels'):
            np.asarray(cache.meta['target'])
        designs = forecast_designs(cache, rows, 9, 16, [6], [81], [81])
        designs['T2'] = transition_tabular_design(cache, rows, 9, [81], designs['D5_CROSS'])
        own, parent = transition_inputs(cache, rows, 9, [81])
        if expected is None:
            expected = designs, own, parent
        else:
            for name, design in designs.items():
                assert design.names == expected[0][name].names
                bits_equal(design.values, expected[0][name].values)
            bits_equal(own, expected[1]); bits_equal(parent, expected[2])
    assert len(variants[-1].forecast_rows(rows, 16)) < len(rows)


def test_saved_parity_rejects_missing_nonfinite_and_wrong_predictions():
    a = np.array([.1, .9], dtype=np.float32)
    assert saved_parity(a, a)['saved_bitwise_equal']
    for b in (a[:1], a + .01, np.array([np.nan, .9], dtype=np.float32)):
        with pytest.raises(AssertionError):
            saved_parity(a, b)
    with pytest.raises(AssertionError):
        bits_equal(np.array([0.], dtype=np.float32), np.array([-0.], dtype=np.float32))
