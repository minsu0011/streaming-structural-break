import numpy as np
import pytest
from src.next7.normalizer_transport import historical_parameters, paired_risk_correlations, COVARIATES
from src.next7.scale import ScaleState, historical_blocks


def test_scale_affine_identity_uses_preclip_values_for_same_raw_blocks():
    t = np.arange(2048)
    h = (np.sin(.731 * t) + .3 * np.cos(.193 * t)).astype(np.float32)
    parameters = historical_parameters(h, 1.9975371345973232)
    prefix = parameters['prefix_length']
    p, f = ScaleState(h[:prefix]), ScaleState(h)
    # Use the same physical log-MAD value and the original positive floors.
    raw = historical_blocks(h[prefix:], max(p.args[2][2], f.args[2][2]))
    old_z = (raw - p.args[2][0]) / p.args[2][1]
    new_z = (raw - f.args[2][0]) / f.args[2][1]
    scale_ratio = np.exp(parameters['covariates'][9 + 8])
    shift = parameters['covariates'][8]
    np.testing.assert_allclose(new_z, old_z * scale_ratio + shift, rtol=0, atol=1e-14)
    assert len(parameters['covariates']) == len(COVARIATES)


def test_correlations_match_materialized_duplicate_seed_rows_and_columns():
    x = np.array([[0., 1.], [2., -1.], [3., 4.], [1., 0.]])
    decisions = np.array([[0, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1], [1, 1, 1, 1]])
    counts = np.array([[1, 1, 1, 1], [2, 0, 1, 1], [0, 1, 1, 2]])
    actual = paired_risk_correlations(x, decisions, counts)
    for draw, count in enumerate(counts):
        selected = np.repeat(np.arange(4), count)
        risk = decisions[np.ix_(selected, selected)].mean(axis=1)
        for column in range(2):
            expected = np.corrcoef(x[selected, column], risk)[0, 1]
            np.testing.assert_allclose(actual[draw, column], expected, rtol=0, atol=2e-14)


def test_zero_variation_is_undefined_and_not_imputed_as_zero_correlation():
    x = np.column_stack((np.ones(3), np.arange(3)))
    result = paired_risk_correlations(x, np.eye(3), np.array([[1, 1, 1], [2, 1, 0]]))
    assert np.isnan(result[:, 0]).all()
    assert np.isnan(result[0]).all()
    assert np.isfinite(result[1, 1])


def test_missing_covariates_and_unsupported_H_are_rejected():
    with pytest.raises(ValueError):
        historical_parameters(np.ones(512), 2.)
    with pytest.raises(ValueError):
        paired_risk_correlations(np.array([[0.], [np.nan]]), np.eye(2), np.ones((1, 2)))
