"""Independent integral reference for the nonzero-correlation Bayes summary."""
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from src.next.channel_evidence import make_channel_args
from src.next.serial_channel_evidence import sum_variances, serial_channel_step


def test_serial_bayes_matches_integrated_normal_mean_prior():
    # Check the stated approximate model itself, not a claim that energy
    # channels are Gaussian or that the resulting alarm is time-uniform.
    points = np.column_stack((.35*np.cos(np.arange(31)*.37)+.2,
                              .3*np.sin(np.arange(31)*.21)-.15))
    correlations = [-.4, .6]
    state = make_channel_args(2, 128)
    variances = sum_variances(state[0], correlations)
    for point in points:
        actual = serial_channel_step(point, state, variances)
    for channel, correlation in enumerate(correlations):
        for prior_variance, column in ((.25, 9), (1., 10)):
            ratios, weights = [], []
            for age, width in zip(state[0], state[1]):
                if age > len(points):
                    break
                total = points[-age:, channel].sum()
                lag = np.abs(np.arange(age)[:, None]-np.arange(age))
                standard_error = np.sqrt(np.sum(correlation**lag))
                null_density = norm.pdf(total, scale=standard_error)
                marginal, error = quad(lambda mean: norm.pdf(total, loc=age*mean,
                    scale=standard_error)*norm.pdf(mean, scale=np.sqrt(prior_variance)),
                    -np.inf, np.inf, epsabs=1e-11, epsrel=1e-11)
                assert error < 1e-9
                ratios.append(marginal/null_density)
                weights.append(width)
            expected = np.log(np.average(ratios, weights=weights))
            np.testing.assert_allclose(actual[channel*13+column], expected,
                                       rtol=1e-6, atol=1e-7)
