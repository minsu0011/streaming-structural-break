"""Historical-only descriptors and separate post-hoc mechanism signatures."""
import numpy as np
from src.next.whitening import fit_historical


def acf(values, lag):
    z = np.asarray(values, dtype=np.float64)
    if len(z) <= lag+2:
        return 0.
    left, right = z[:-lag], z[lag:]
    left, right = left-left.mean(), right-right.mean()
    return float(np.dot(left, right)/max(np.sqrt(np.dot(left, left)*np.dot(right, right)), 1e-12))


def historical_descriptors(historical):
    h = np.asarray(historical, dtype=np.float64)
    mean, median = float(h.mean()), float(np.median(h))
    sd, mad = max(float(h.std()), 1e-8), max(float(1.4826*np.median(np.abs(h-median))), 1e-8)
    z = (h-mean)/sd
    rz = (h-median)/mad
    robust = np.clip(rz, -12., 12.)
    result = {'location_mean': mean, 'location_median': median, 'log_sd': float(np.log(sd)),
              'log_mad': float(np.log(mad)), 'skewness': float(np.mean(z**3)),
              'log_kurtosis': float(np.log(max(float(np.mean(z**4)), 1e-8))),
              'tail2_rate': float(np.mean(np.abs(rz)>2)), 'tail3_rate': float(np.mean(np.abs(rz)>3)),
              'log_historical_length': float(np.log(len(h)))}
    for lag in [1, 2, 4, 8]:
        result[f'acf_{lag}'] = acf(robust, lag)
    for order in [4, 8]:
        fit = fit_historical(h, order)
        residual = fit['historical_innovations']
        for j, coefficient in enumerate(fit['coefficients']):
            result[f'ar{order}_coefficient_{j+1}'] = float(coefficient)
        result[f'ar{order}_log_innovation_variance'] = float(np.log(max(float(np.var(residual)), 1e-12)))
        result[f'ar{order}_residual_acf1'] = acf(residual, 1)
        result[f'ar{order}_squared_residual_acf1'] = acf(np.minimum(residual**2, np.quantile(residual**2, .99)), 1)
        result[f'ar{order}_squared_residual_acf2'] = acf(np.minimum(residual**2, np.quantile(residual**2, .99)), 2)
    fft = np.abs(np.fft.rfft(robust-robust.mean()))**2
    freq = np.fft.rfftfreq(len(robust))
    total = max(float(fft.sum()), 1e-12)
    for j, (a, b) in enumerate([(0, .05), (.05, .15), (.15, .3), (.3, .500001)]):
        result[f'spectral_band_{j}'] = float(fft[(freq >= a) & (freq < b)].sum()/total)
    return result


def mechanism_signature(historical, online, tau, policy):
    """Evaluation only. The causal feature engine must never call this function."""
    names = ['MEAN_DOMINANT', 'SCALE_DOMINANT', 'DEPENDENCE_DOMINANT', 'TAIL_DOMINANT']
    result = {'signature': 'NO_BREAK_CONTROL', 'pre_count': 0, 'post_count': 0,
              'mean_effect': 0., 'scale_effect': 0., 'dependence_effect': 0., 'tail_effect': 0.}
    if tau < 0:
        return result
    window = policy['window']
    pre = np.concatenate([np.asarray(historical, dtype=np.float64), np.asarray(online[:tau], dtype=np.float64)])[-window:]
    post = np.asarray(online[tau:tau+window], dtype=np.float64)
    result.update(pre_count=len(pre), post_count=len(post))
    if min(len(pre), len(post)) < policy['minimum_each_side']:
        result['signature'] = 'WEAK_OR_AMBIGUOUS'
        return result
    mean, sd = float(pre.mean()), max(float(pre.std()), 1e-8)
    median, mad = float(np.median(pre)), max(float(1.4826*np.median(np.abs(pre-np.median(pre)))), sd*.01, 1e-8)
    effects = [abs(float(post.mean())-mean)/sd,
               abs(float(np.log(max(float(post.std()), 1e-8)/sd))),
               max(abs(acf(post, lag)-acf(pre, lag)) for lag in policy['acf_lags']),
               abs(float(np.mean(np.abs((post-median)/mad) > policy['tail_cutoff_pre_robust_scale']))-
                   float(np.mean(np.abs((pre-median)/mad) > policy['tail_cutoff_pre_robust_scale'])))]
    result.update(zip(['mean_effect', 'scale_effect', 'dependence_effect', 'tail_effect'], effects))
    effects = np.asarray(effects)/np.asarray([policy['mean_effect_threshold'], policy['absolute_log_sd_ratio_threshold'],
        policy['max_acf_difference_threshold'], policy['tail_exceedance_rate_difference_threshold']])
    order = np.argsort(-effects, kind='stable')
    if effects[order[0]] < 1:
        result['signature'] = 'WEAK_OR_AMBIGUOUS'
    elif effects[order[1]] >= 1 and effects[order[0]] < 1.5*effects[order[1]]:
        result['signature'] = 'MIXED'
    else:
        result['signature'] = names[order[0]]
    return result
