"""H-fitted AR(1) sum-variance correction for the fixed GLR/Bayes age grid.

For unit marginal variance and AR(1) correlation rho, a sum of n samples has
variance v_n = n + 2 sum_{k=1}^{n-1}(n-k)rho**k. The normal mean-prior Bayes
summary treats that sum as N(n*mu, v_n); it is an approximation for these
non-Gaussian energy channels, not a calibrated sequential likelihood ratio.
"""
from pathlib import Path
import hashlib
import numpy as np
from numba import njit
from src.next.channel_evidence import CHANNEL_STATS, _logadd


def sum_variances(ages, correlations):
    ages = np.asarray(ages, dtype=np.int64)
    rho = np.asarray(correlations, dtype=float)
    if ages.ndim != 1 or rho.ndim != 1 or not len(ages) or np.any(ages < 1) or not np.isfinite(rho).all() or np.any(np.abs(rho) >= 1):
        raise ValueError('Positive ages and stationary AR(1) correlations required')
    result = np.empty((len(rho), len(ages)))
    for channel, value in enumerate(rho):
        for k, age in enumerate(ages):
            lags = np.arange(1, age)
            result[channel, k] = age+2*np.sum((age-lags)*value**lags)
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise RuntimeError('AR(1) sum variance must be strictly positive')
    return result


def historical_correlations(channels):
    values = np.clip(np.asarray(channels, dtype=float), -12., 12.)
    if values.ndim != 2 or len(values) < 16 or not np.isfinite(values).all():
        raise ValueError('Finite historical channel matrix with at least 16 points required')
    left, right = values[:-1].copy(), values[1:].copy()
    left -= left.mean(axis=0)
    right -= right.mean(axis=0)
    denominator = np.sqrt((left*left).sum(axis=0)*(right*right).sum(axis=0))
    rho = np.divide((left*right).sum(axis=0), denominator, out=np.zeros(values.shape[1]), where=denominator > 1e-12)
    return np.clip(rho, -.95, .95)


@njit(cache=True)
def serial_channel_step(values, args, age_variances):
    ages, widths, ewma, cusum, cumulative, prefix, counter, memory, output = args
    t = counter[0]+1
    halves = (8., 32., 128.)
    for channel in range(len(values)):
        value = min(max(values[channel], -12.), 12.)
        cursor = channel*len(CHANNEL_STATS)
        output[cursor] = value
        for k in range(3):
            alpha = 1.-np.exp(-np.log(2.)/halves[k])
            ewma[channel, k] = (1.-alpha)*ewma[channel, k]+alpha*value
            variance = max(alpha/(2.-alpha)*(1.-(1.-alpha)**(2*t)), 1e-12)
            output[cursor+1+k] = ewma[channel, k]/np.sqrt(variance)
        cusum[channel, 0] = max(0., cusum[channel, 0]+value-.25)
        cusum[channel, 1] = max(0., cusum[channel, 1]-value-.25)
        output[cursor+4] = np.log1p(cusum[channel, 0])
        output[cursor+5] = np.log1p(cusum[channel, 1])
        cumulative[channel] += value
        maximum, age_at_max = 0., 1
        mix, b025, b1 = -1e300, -1e300, -1e300
        count, width = 0, 0.
        for k in range(len(ages)):
            age = ages[k]
            if age > t:
                break
            count += 1
            width += widths[k]
            total = cumulative[channel]-prefix[(t-age)%len(prefix), channel]
            energy = total*total
            variance = age_variances[channel, k]
            ratio = age/variance
            glr = .5*energy/variance
            if glr > maximum:
                maximum, age_at_max = glr, age
            mix = _logadd(mix, glr)
            b025 = _logadd(b025, np.log(widths[k])-.5*np.log1p(.25*age*ratio)+.5*.25*ratio*ratio*energy/(1.+.25*age*ratio))
            b1 = _logadd(b1, np.log(widths[k])-.5*np.log1p(age*ratio)+.5*ratio*ratio*energy/(1.+age*ratio))
        prefix[t%len(prefix), channel] = cumulative[channel]
        memory[channel, 0] = max(memory[channel, 0], maximum)
        memory[channel, 1] = max(np.exp(-np.log(2.)/32.)*memory[channel, 1], maximum)
        output[cursor+6] = maximum
        output[cursor+7] = mix-np.log(count)
        output[cursor+8] = np.log2(age_at_max)
        output[cursor+9] = b025-np.log(width)
        output[cursor+10] = b1-np.log(width)
        output[cursor+11] = memory[channel, 0]
        output[cursor+12] = memory[channel, 1]
    counter[0] = t
    return output


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
