"""Fixed-grid causal location evidence for a small vector of calibrated scores."""
from pathlib import Path
import hashlib
import numpy as np
from numba import njit


CHANNEL_STATS = ('current', 'ewma8', 'ewma32', 'ewma128', 'cusum_positive', 'cusum_negative',
    'glr_max', 'glr_logmean', 'glr_logage', 'bayes025', 'bayes1', 'memory_max', 'memory_decay')


@njit(cache=True)
def _logadd(left, right):
    maximum = max(left, right)
    return maximum + np.log(np.exp(left-maximum) + np.exp(right-maximum))


@njit(cache=True)
def channel_step(values, args):
    ages, widths, ewma, cusum, cumulative, prefix, counter, memory, output = args
    t = counter[0] + 1
    halves = (8.,32.,128.)
    for channel in range(len(values)):
        value = min(max(values[channel], -12.), 12.)
        cursor = channel * len(CHANNEL_STATS)
        output[cursor] = value
        for k in range(3):
            alpha = 1. - np.exp(-np.log(2.) / halves[k])
            ewma[channel,k] = (1.-alpha)*ewma[channel,k] + alpha*value
            variance = max(alpha/(2.-alpha)*(1.-(1.-alpha)**(2*t)), 1e-12)
            output[cursor+1+k] = ewma[channel,k]/np.sqrt(variance)
        cusum[channel,0] = max(0.,cusum[channel,0]+value-.25)
        cusum[channel,1] = max(0.,cusum[channel,1]-value-.25)
        output[cursor+4] = np.log1p(cusum[channel,0])
        output[cursor+5] = np.log1p(cusum[channel,1])
        cumulative[channel] += value
        maximum, age_at_max = 0.,1
        mix, b025, b1 = -1e300,-1e300,-1e300
        count, width = 0,0.
        for k in range(len(ages)):
            age = ages[k]
            if age > t:
                break
            count += 1
            width += widths[k]
            total = cumulative[channel]-prefix[(t-age)%len(prefix),channel]
            energy = total*total
            glr = .5*energy/age
            if glr > maximum:
                maximum, age_at_max = glr, age
            mix = _logadd(mix, glr)
            b025 = _logadd(b025, np.log(widths[k])-.5*np.log1p(.25*age)+.5*.25*energy/(1.+.25*age))
            b1 = _logadd(b1, np.log(widths[k])-.5*np.log1p(age)+.5*energy/(1.+age))
        prefix[t%len(prefix),channel] = cumulative[channel]
        memory[channel,0] = max(memory[channel,0],maximum)
        memory[channel,1] = max(np.exp(-np.log(2.)/32.)*memory[channel,1],maximum)
        output[cursor+6] = maximum
        output[cursor+7] = mix-np.log(count)
        output[cursor+8] = np.log2(age_at_max)
        output[cursor+9] = b025-np.log(width)
        output[cursor+10] = b1-np.log(width)
        output[cursor+11] = memory[channel,0]
        output[cursor+12] = memory[channel,1]
    counter[0] = t
    return output


@njit(cache=True)
def replay_channels(values, args):
    output = np.empty((len(values),len(args[-1])),dtype=np.float32)
    for j in range(len(values)):
        output[j] = channel_step(values[j],args)
    return output


def make_channel_args(channels, max_age=128):
    if channels < 1 or channels > 16 or max_age not in (128,512):
        raise ValueError('Unsupported channel count or fixed age grid')
    ages = np.asarray([2**k for k in range(max_age.bit_length())],dtype=np.int64)
    return (ages,np.diff(np.r_[0,ages]).astype(float),np.zeros((channels,3)),np.zeros((channels,2)),
        np.zeros(channels),np.zeros((max_age+1,channels)),np.zeros(1,dtype=np.int64),np.zeros((channels,2)),
        np.empty(channels*len(CHANNEL_STATS),dtype=np.float32))


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
