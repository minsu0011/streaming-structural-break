import numpy as np
import pytest
from scipy.special import logsumexp
from src.next.engine import SequentialState, EngineConfig, FEATURE_NAMES, FEATURE_GROUPS, AGES, AGE_WIDTHS, MEAN_AMPLITUDES, VARIANCE_RATIOS


@pytest.mark.parametrize('normalization', ['sd','mad','arch1','arch2','ewma','uniform','gaussian'])
def test_causal_prefix_and_stream_batch_exact(normalization):
    rng = np.random.default_rng(793)
    h = rng.standard_t(4, 1200).astype(np.float32)
    prefix = rng.normal(size=173).astype(np.float32)
    suffix = rng.normal(2, 3, 220).astype(np.float32)
    config = EngineConfig(normalization=normalization)
    online = SequentialState(h, config)
    one = np.array([online.update(x).copy() for x in prefix])
    two = SequentialState(h, config).replay(np.r_[prefix, suffix])
    np.testing.assert_array_equal(one, two[:len(prefix)])
    assert np.isfinite(two).all()
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES) == len(FEATURE_GROUPS) == two.shape[1]


def test_glr_and_bayes_ring_match_direct_suffix_likelihoods():
    rng = np.random.default_rng(399)
    h = rng.normal(size=1500).astype(np.float32)
    points = rng.normal(.2, 1.1, 350).astype(np.float32)
    state = SequentialState(h)
    moments = state.args[8].copy()
    features = state.replay(points)
    # Output stores float32 z; use causal filter directly for double precision
    # comparison so rounding in a diagnostic column cannot alter the oracle.
    from src.next.whitening import innovation_update
    from src.next.normalization import normalization_update
    oracle = SequentialState(h)
    a = oracle.args
    w = []
    for point in points:
        residual = innovation_update(point, a[0], a[1], a[2], a[3])
        z, _ = normalization_update(residual, a[4], a[5], a[6], a[7][0])
        w.append((z-moments[0])/moments[1])
    w = np.asarray(w)
    for t in [1,2,4,17,128,129,130,257,350]:
        ages = AGES[AGES <= t]
        sums = np.array([w[t-age:t].sum() for age in ages])
        squares = np.array([np.square(w[t-age:t]).sum() for age in ages])
        mean_lr = sums*sums/(2*ages)
        second = np.maximum(squares/ages, 1e-8)
        variance_lr = .5*ages*(second-1-np.log(second))
        weights = AGE_WIDTHS[:len(ages)]
        logprior = np.log(weights/weights.sum())
        bf025 = -.5*np.log1p(ages*.25)+.5*.25*sums*sums/(1+ages*.25)
        mean_bfs = np.array([logsumexp(MEAN_AMPLITUDES*s-.5*n*MEAN_AMPLITUDES**2)-np.log(6) for n,s in zip(ages,sums)])
        variance_bfs = np.array([logsumexp(-.5*n*np.log(VARIANCE_RATIOS)+.5*s*(1-1/VARIANCE_RATIOS))-np.log(4) for n,s in zip(ages,squares)])
        expected = {'glr_mean_max': mean_lr.max(), 'glr_variance_max': variance_lr.max(),
                    'bayes_mean_prior025': logsumexp(logprior+bf025),
                    'bayes_mean_amplitude_mixture': logsumexp(logprior+mean_bfs),
                    'bayes_variance_mixture': logsumexp(logprior+variance_bfs)}
        for name, value in expected.items():
            np.testing.assert_allclose(features[t-1, FEATURE_NAMES.index(name)], np.float32(value), rtol=1e-6, atol=2e-6)
    assert features[0, FEATURE_NAMES.index('glr_joint_max')] == 0


def test_magnitude_dependence_and_ecdf_state_stay_bounded():
    rng = np.random.default_rng(389)
    h = rng.normal(size=1000).astype(np.float32)
    state = SequentialState(h, EngineConfig(cdf_bins=8))
    before = state.state_array_bytes
    output = state.replay(np.tile(rng.normal(size=100).astype(np.float32), 200))
    assert state.state_array_bytes == before
    assert np.isfinite(output).all()
    np.testing.assert_allclose(state.args[14].sum(axis=1), np.ones(3), atol=1e-12, rtol=0)
    np.testing.assert_array_equal(output[:,FEATURE_NAMES.index('evidence_cumulative_max')],
                                  np.maximum.accumulate(output[:,FEATURE_NAMES.index('evidence_current')]))


def test_frozen_historical_parameters_are_unchanged_online():
    rng = np.random.default_rng(899)
    state = SequentialState(rng.normal(size=1000), EngineConfig(normalization='gaussian'))
    protected = [0,1,4,6,8,9,13,15]
    copies = {j: state.args[j].copy() for j in protected}
    state.replay(rng.normal(size=300))
    for j in protected:
        np.testing.assert_array_equal(state.args[j], copies[j])
