"""Causal innovation evidence engine with bounded state and fixed small grids."""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import numpy as np
from numba import njit
from src.next.whitening import fit_historical, innovation_update
from src.next.normalization import fit_normalization, normalization_update

HALF_LIVES = np.array([8., 32., 128.])
ALPHAS = 1.-np.exp(-np.log(2.)/HALF_LIVES)
DRIFTS = np.array([.1, .25, .5])
LAGS = np.array([1, 2, 4, 8])
AGES = np.array([1, 2, 4, 8, 16, 32, 64, 128])
AGE_WIDTHS = np.array([1., 1., 2., 4., 8., 16., 32., 64.])
MEAN_AMPLITUDES = np.array([-.25, .25, -.5, .5, -1., 1.])
VARIANCE_RATIOS = np.array([.5, .75, 1.5, 2.])


@dataclass(frozen=True)
class EngineConfig:
    order: int = 8
    normalization: str = 'sd'
    cdf_bins: int = 16


def feature_contract():
    names, groups = [], []
    def add(group, entries):
        names.extend(entries); groups.extend([group]*len(entries))
    add('I', ['current_z', 'current_abs_z', 'current_centered_square', 'current_log_energy', 'elapsed_log1p'])
    for half in [8,32,128]:
        add('S1', [f'mean_ewma_signed_{half}', f'mean_ewma_abs_{half}'])
    for drift in ['010','025','050']:
        add('S1', [f'mean_cusum_positive_{drift}', f'mean_cusum_negative_{drift}'])
    add('S1', ['page_hinkley_positive', 'page_hinkley_negative'])
    for half in [8,32,128]:
        add('S2', [f'log_second_ratio_{half}', f'log_variance_ratio_{half}', f'absolute_moment_delta_{half}', f'log_energy_delta_{half}'])
    for drift in ['010','025','050']:
        add('S2', [f'scale_cusum_positive_{drift}', f'scale_cusum_negative_{drift}'])
    for half in [8,32,128]:
        for kind in ['signed','absolute','squared']:
            add('S3', [f'{kind}_innovation_product_lag{lag}_{half}' for lag in [1,2,4,8]])
    for half in [8,32,128]:
        add('S4', [f'cdf_cvm_{half}', f'cdf_ad_{half}', f'lower_tail_delta_{half}', f'upper_tail_delta_{half}', f'central_mass_delta_{half}'])
    add('G', ['glr_mean_max', 'glr_variance_max', 'glr_joint_max', 'glr_mean_log_average', 'glr_variance_log_average',
              'glr_mean_argmax_log_age', 'glr_variance_argmax_log_age', 'glr_max_multiplicity_corrected'])
    add('B', ['bayes_mean_prior025', 'bayes_mean_prior1', 'bayes_mean_amplitude_mixture', 'bayes_variance_mixture',
              'bayes_joint_mixture', 'bayes_mixture_bounded_score'])
    add('M', ['evidence_current', 'evidence_cumulative_max', 'evidence_decayed_max', 'evidence_leaky_accumulator',
              'evidence_posterior_like_memory', 'evidence_hysteresis'])
    return tuple(names), tuple(groups)


FEATURE_NAMES, FEATURE_GROUPS = feature_contract()
N_FEATURES = len(FEATURE_NAMES)


def feature_hash(config):
    directory = Path(__file__).parent
    body = b''.join((directory/name).read_bytes() for name in ['engine.py','whitening.py','normalization.py'])
    return hashlib.sha256(body+json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()


@njit(cache=True)
def _logadd(a, b):
    maximum = max(a, b)
    return maximum+np.log(np.exp(a-maximum)+np.exp(b-maximum))


@njit(cache=True)
def _step(point, args):
    (ar_ref, ar_coef, ar_ring, ar_counter, norm_parameters, conditional, sorted_h,
     integers, moments, dep_reference, z_ring, ewma, dep_ewma, bin_reference, bin_ewma,
     tail_reference, tail_ewma, cusums, ph, cumulative, prefix, memories, out) = args
    residual = innovation_update(point, ar_ref, ar_coef, ar_ring, ar_counter)
    z, u = normalization_update(residual, norm_parameters, conditional, sorted_h, integers[0])
    t = integers[2]+1
    w = (z-moments[0])/moments[1]
    square = w*w
    absolute = abs(w)
    log_energy = np.log(square+1e-4)
    out[0] = z
    out[1] = abs(z)
    out[2] = square-1.
    out[3] = log_energy-moments[3]
    out[4] = np.log1p(t)
    cursor = 5
    evidence = 0.
    for j in range(3):
        alpha = ALPHAS[j]
        ewma[j,0] = (1.-alpha)*ewma[j,0]+alpha*w
        ewma[j,1] = (1.-alpha)*ewma[j,1]+alpha*square
        ewma[j,2] = (1.-alpha)*ewma[j,2]+alpha*absolute
        ewma[j,3] = (1.-alpha)*ewma[j,3]+alpha*log_energy
        noise = np.sqrt(max(alpha/(2.-alpha)*(1.-(1.-alpha)**(2*t)), 1e-10))
        value = ewma[j,0]/noise
        out[cursor] = value
        out[cursor+1] = abs(value)
        evidence = max(evidence, abs(value))
        cursor += 2
    for j in range(3):
        cusums[0,j,0] = max(0., cusums[0,j,0]+w-DRIFTS[j])
        cusums[0,j,1] = max(0., cusums[0,j,1]-w-DRIFTS[j])
        out[cursor] = np.log1p(cusums[0,j,0])
        out[cursor+1] = np.log1p(cusums[0,j,1])
        cursor += 2
    ph[4] += (w-ph[4])/t
    ph[0] += w-ph[4]-.1
    ph[1] = min(ph[1], ph[0])
    ph[2] += -w+ph[4]-.1
    ph[3] = min(ph[3], ph[2])
    out[cursor] = np.log1p(ph[0]-ph[1])
    out[cursor+1] = np.log1p(ph[2]-ph[3])
    cursor += 2
    for j in range(3):
        alpha = ALPHAS[j]
        noise = np.sqrt(max(alpha/(2.-alpha)*(1.-(1.-alpha)**(2*t)), 1e-10))
        out[cursor] = np.log(max(ewma[j,1], 1e-8))
        out[cursor+1] = np.log(max(ewma[j,1]-ewma[j,0]**2, 1e-8))
        out[cursor+2] = (ewma[j,2]-moments[2])/(moments[5]*noise)
        out[cursor+3] = (ewma[j,3]-moments[3])/(moments[6]*noise)
        evidence = max(evidence, abs(out[cursor+2]), abs(out[cursor+3]))
        cursor += 4
    scaled_square = (square-1.)/moments[4]
    for j in range(3):
        cusums[1,j,0] = max(0., cusums[1,j,0]+scaled_square-DRIFTS[j])
        cusums[1,j,1] = max(0., cusums[1,j,1]-scaled_square-DRIFTS[j])
        out[cursor] = np.log1p(cusums[1,j,0])
        out[cursor+1] = np.log1p(cusums[1,j,1])
        cursor += 2
    current = (w, absolute-moments[2], square-1.)
    for j in range(3):
        alpha = ALPHAS[j]
        noise = np.sqrt(max(alpha/(2.-alpha)*(1.-(1.-alpha)**(2*t)), 1e-10))
        for kind in range(3):
            for k in range(4):
                product = current[kind]*z_ring[kind, (integers[2]-LAGS[k]) % 8]
                dep_ewma[j,kind,k] = (1.-alpha)*dep_ewma[j,kind,k]+alpha*product
                value = (dep_ewma[j,kind,k]-dep_reference[kind,k,0])/(dep_reference[kind,k,1]*noise)
                out[cursor] = value
                if kind == 0:
                    evidence = max(evidence, abs(value))
                cursor += 1
    for kind in range(3):
        z_ring[kind, integers[2] % 8] = current[kind]
    bins = integers[1]
    occupied = min(int(u*bins), bins-1)
    for j in range(3):
        alpha = ALPHAS[j]
        null_variance = max(alpha/(2.-alpha)*(1.-(1.-alpha)**(2*t)), 1e-10)
        cumulative_difference = 0.
        null_cdf = 0.
        cvm = 0.
        ad = 0.
        for k in range(bins):
            bin_ewma[j,k] = (1.-alpha)*bin_ewma[j,k]+alpha*(occupied == k)
            cumulative_difference += bin_ewma[j,k]-bin_reference[k]
            null_cdf += bin_reference[k]
            if k < bins-1:
                cvm += cumulative_difference*cumulative_difference
                ad += cumulative_difference*cumulative_difference/max(null_cdf*(1.-null_cdf), .01)
        out[cursor] = cvm/(bins*null_variance)
        out[cursor+1] = ad/(bins*null_variance)
        indicators = (u < .05, u > .95, .25 <= u <= .75)
        for k in range(3):
            tail_ewma[j,k] = (1.-alpha)*tail_ewma[j,k]+alpha*indicators[k]
            out[cursor+2+k] = (tail_ewma[j,k]-tail_reference[k])/np.sqrt(max(tail_reference[k]*(1.-tail_reference[k])*null_variance, 1e-8))
        cursor += 5
    cumulative[0] += w
    cumulative[1] += square
    max_mean, max_variance, max_joint = 0., 0., 0.
    mean_age, variance_age = 1., 1.
    glr_mean_mix, glr_variance_mix = -1e300, -1e300
    bayes025, bayes1, bayes_amplitude, bayes_variance = -1e300, -1e300, -1e300, -1e300
    valid_ages = 0
    width_total = 0.
    for age_index in range(8):
        age = AGES[age_index]
        if age > t:
            break
        valid_ages += 1
        width_total += AGE_WIDTHS[age_index]
        total = cumulative[0]-prefix[0, (t-age) % 129]
        total_squared = cumulative[1]-prefix[1, (t-age) % 129]
        mean = total/age
        second = max(total_squared/age, 1e-8)
        variance = max(second-mean*mean, 1e-8)
        mean_lr = total*total/(2.*age)
        variance_lr = .5*age*(second-1.-np.log(second))
        # Joint mean/variance maximum likelihood is singular at one point.
        # The joint statistic has a fixed minimum support of four observations.
        joint_lr = .5*age*(second-1.-np.log(variance)) if age >= 4 else 0.
        if mean_lr > max_mean:
            max_mean = mean_lr; mean_age = age
        if variance_lr > max_variance:
            max_variance = variance_lr; variance_age = age
        max_joint = max(max_joint, joint_lr)
        glr_mean_mix = _logadd(glr_mean_mix, mean_lr)
        glr_variance_mix = _logadd(glr_variance_mix, variance_lr)
        log_width = np.log(AGE_WIDTHS[age_index])
        bf025 = -.5*np.log1p(age*.25)+.5*.25*total*total/(1.+age*.25)
        bf1 = -.5*np.log1p(age)+.5*total*total/(1.+age)
        mean_grid = -1e300
        for amplitude in MEAN_AMPLITUDES:
            mean_grid = _logadd(mean_grid, amplitude*total-.5*age*amplitude*amplitude)
        variance_grid = -1e300
        for ratio in VARIANCE_RATIOS:
            variance_grid = _logadd(variance_grid, -.5*age*np.log(ratio)+.5*total_squared*(1.-1./ratio))
        bayes025 = _logadd(bayes025, log_width+bf025)
        bayes1 = _logadd(bayes1, log_width+bf1)
        bayes_amplitude = _logadd(bayes_amplitude, log_width+mean_grid-np.log(6.))
        bayes_variance = _logadd(bayes_variance, log_width+variance_grid-np.log(4.))
    prefix[0, t % 129] = cumulative[0]
    prefix[1, t % 129] = cumulative[1]
    out[cursor] = max_mean
    out[cursor+1] = max_variance
    out[cursor+2] = max_joint
    out[cursor+3] = glr_mean_mix-np.log(valid_ages)
    out[cursor+4] = glr_variance_mix-np.log(valid_ages)
    out[cursor+5] = np.log2(mean_age)
    out[cursor+6] = np.log2(variance_age)
    out[cursor+7] = max(max_mean, max_variance)-np.log(2.*valid_ages)
    cursor += 8
    bayes025 -= np.log(width_total)
    bayes1 -= np.log(width_total)
    bayes_amplitude -= np.log(width_total)
    bayes_variance -= np.log(width_total)
    joint_bayes = _logadd(bayes_amplitude, bayes_variance)-np.log(2.)
    out[cursor] = bayes025
    out[cursor+1] = bayes1
    out[cursor+2] = bayes_amplitude
    out[cursor+3] = bayes_variance
    out[cursor+4] = joint_bayes
    out[cursor+5] = 1./(1.+np.exp(-min(max(joint_bayes, -30.), 30.)))
    cursor += 6
    memories[0] = evidence
    memories[1] = max(memories[1], evidence)
    memories[2] = max(np.exp(-np.log(2.)/32.)*memories[2], evidence)
    memories[3] = max(0., np.exp(-np.log(2.)/32.)*memories[3]+evidence-1.)
    memories[4] = min(max(.98*memories[4]+joint_bayes, -30.), 30.)
    if evidence >= 3.:
        memories[5] = 1.
    elif evidence < 1.:
        memories[5] = 0.
    for j in range(6):
        out[cursor+j] = memories[j]
    integers[2] = t
    return out


@njit(cache=True)
def _replay(points, args):
    result = np.empty((len(points), N_FEATURES), dtype=np.float32)
    for j in range(len(points)):
        result[j] = _step(points[j], args)
    return result


class SequentialState:
    def __init__(self, historical, config=EngineConfig()):
        if config.cdf_bins not in (8,16):
            raise ValueError('CDF bin count outside frozen small grid')
        self.config = config
        ar = fit_historical(historical, config.order)
        norm = fit_normalization(ar['historical_innovations'], config.normalization)
        z, u = norm['historical_z'], norm['historical_uniform']
        mu, sd = float(z.mean()), max(float(z.std()), 1e-6)
        w = (z-mu)/sd
        absolute, square = np.abs(w), w*w
        log_energy = np.log(square+1e-4)
        moments = np.array([mu, sd, absolute.mean(), log_energy.mean(), max(float(square.std()),1e-4),
                            max(float(absolute.std()),1e-4), max(float(log_energy.std()),1e-4)])
        values = np.asarray([w, absolute-moments[2], square-1.])
        dep_reference = np.empty((3,4,2), dtype=np.float64)
        for kind in range(3):
            for k, lag in enumerate(LAGS):
                products = values[kind,lag:]*values[kind,:-lag]
                dep_reference[kind,k] = [products.mean(), max(float(products.std()),1e-4)]
        ewma = np.tile(np.array([0.,1.,moments[2],moments[3]]), (3,1))
        dep_ewma = np.tile(dep_reference[:,:,0], (3,1,1))
        occupied = np.minimum((u*config.cdf_bins).astype(int), config.cdf_bins-1)
        bin_reference = np.bincount(occupied, minlength=config.cdf_bins)/len(u)
        tail_reference = np.array([(u < .05).mean(), (u > .95).mean(), ((u >= .25) & (u <= .75)).mean()])
        self.args = (ar['reference'], ar['coefficients'], ar['ring'], ar['counter'],
            norm['parameters'], norm['conditional'], norm['sorted_h'], np.array([norm['mode'],config.cdf_bins,0], dtype=np.int64),
            moments, dep_reference, values[:,-8:].copy(), ewma, dep_ewma,
            bin_reference, np.tile(bin_reference, (3,1)), tail_reference, np.tile(tail_reference, (3,1)),
            np.zeros((2,3,2)), np.zeros(5), np.zeros(2), np.zeros((2,129)), np.zeros(6), np.empty(N_FEATURES,dtype=np.float32))
        self.historical_pole_projected = ar['pole_projected']

    def update(self, point):
        return _step(float(point), self.args)

    def replay(self, points):
        return _replay(np.asarray(points, dtype=np.float32), self.args)

    @property
    def state_array_bytes(self):
        return sum(a.nbytes for a in self.args)
