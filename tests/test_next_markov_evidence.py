import numpy as np
import pytest
from scipy.special import gammaln, xlogy
from src.next.markov_evidence import MarkovState, MarkovConfig, feature_names


def _integrated_loglik(counts, prior):
    return (gammaln(prior.sum(axis=1)) - gammaln(prior.sum(axis=1) + counts.sum(axis=1))).sum() + (gammaln(prior + counts) - gammaln(prior)).sum()


@pytest.mark.parametrize('strength', [4., 16.])
def test_incremental_dirichlet_bf_and_glr_match_full_recalculation(strength):
    rng = np.random.default_rng(408)
    h = rng.standard_t(5, 1200).astype(np.float32)
    points = rng.normal(size=1100).astype(np.float32)
    state = MarkovState(h, MarkovConfig(prior_strength=strength))
    edges, lags, ages = state.args[:3]
    all_bins = np.searchsorted(edges, np.r_[h, points], side='right')
    for t, point in enumerate(points, 1):
        state.update(point)
        if t not in [1, 2, 8, 129, 511, 512, 513, 520, 1050, 1100]:
            continue
        for li, lag in enumerate(lags):
            beta = state.historical_counts[li] + .5
            p = beta / beta.sum(axis=1, keepdims=True)
            alpha = strength * p
            for ai, age in enumerate(ages):
                indices = np.arange(len(h) + max(0, t - age), len(h) + t)
                counts = np.bincount(4 * all_bins[indices - lag] + all_bins[indices], minlength=16).reshape(4, 4)
                np.testing.assert_array_equal(counts, state.args[9][li, ai])
                expected_bf = _integrated_loglik(counts, alpha) - _integrated_loglik(counts, beta)
                expected_glr = xlogy(counts, counts).sum() - xlogy(counts.sum(axis=1), counts.sum(axis=1)).sum() - (counts * np.log(p)).sum()
                np.testing.assert_allclose(state.args[12][li, ai], expected_bf, atol=2e-10)
                np.testing.assert_allclose(state.args[11][li, ai], expected_glr, atol=2e-10)


@pytest.mark.parametrize('max_age', [128, 512])
def test_markov_exact_prefix_batch_stream_fixed_memory(max_age):
    rng = np.random.default_rng(965)
    h = rng.normal(size=1000).astype(np.float32)
    points = rng.normal(size=2000).astype(np.float32)
    config = MarkovConfig(max_age=max_age)
    state = MarkovState(h, config)
    size = state.state_array_bytes
    historical_arrays = [a.copy() for a in state.args[:8]]
    streaming = np.array([state.update(point).copy() for point in points[:701]])
    batch = MarkovState(h, config).replay(points)
    np.testing.assert_array_equal(streaming, batch[:701])
    assert size == state.state_array_bytes
    for old, new in zip(historical_arrays, state.args[:8]):
        np.testing.assert_array_equal(old, new)
    assert np.isfinite(batch).all()


def test_transition_change_with_unchanged_marginal_distribution():
    rng = np.random.default_rng(698)
    def ar(n, phi):
        result = np.empty(n, dtype=np.float32)
        result[0] = rng.normal()
        for j in range(1, n):
            result[j] = phi * result[j - 1] + np.sqrt(1 - phi**2) * rng.normal()
        return result
    h = ar(6000, .7)
    unchanged = MarkovState(h).replay(ar(1000, .7))
    changed = MarkovState(h).replay(ar(1000, -.7))
    column = feature_names({}).index('transition_lag1_bf_logmixture')
    assert np.median(changed[512:, column]) > np.median(unchanged[512:, column]) + 50


def test_constant_history_has_finite_smoothed_probabilities():
    state = MarkovState(np.ones(100))
    features = state.replay(np.r_[np.ones(100), np.zeros(100)])
    assert np.isfinite(features).all()
