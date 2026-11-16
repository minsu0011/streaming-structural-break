"""Known-truth sampling calibration of the unchanged rank interval methods."""
from itertools import product
import hashlib
import math
import numpy as np
from scipy.special import expit, ndtr
from src.next8.statistics import rank_influence
from src.next8.paired_bootstrap import evaluate_partition


def cells():
    return list(product((64, 256, 1280), (0., .95), (0., .36)))


def sample(groups, rho, shift, replicate, *, length=512):
    assert groups >= 4 and groups % 2 == 0 and 0 <= rho < 1 and length >= 400
    seed = 9101604 + groups * 1009 + replicate * 100003
    rng, target_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
    tau = np.full(groups, 513, dtype=np.int32)
    changed = target_rng.permutation(groups)[:groups // 2]
    tau[changed] = target_rng.integers(1, 400, len(changed))
    y = (np.arange(length)[None, :] >= tau[:, None]).astype(np.uint8)
    z = np.empty((groups, length), dtype=np.float64)
    previous = rng.normal(size=groups)
    for t in range(length):
        previous = rho * previous + np.sqrt(1. - rho * rho) * rng.normal(size=groups)
        z[:, t] = previous
    probability = expit(z + shift * y)
    return (y.reshape(-1), probability.reshape(-1), np.tile(np.arange(length), groups),
            np.repeat(np.arange(700000, 700000 + groups), length), float(ndtr(shift / np.sqrt(2.))))


def interval_record(low, high, truth):
    if low is None or high is None:
        return {'status': 'INSUFFICIENT_BOOTSTRAP_SUPPORT', 'ci_low': None, 'ci_high': None,
                'covers_truth': None, 'width': None, 'lower_gt_0_505': None, 'lower_gt_0_47': None}
    assert math.isfinite(low) and math.isfinite(high) and 0 <= low <= high <= 1
    return {'status': 'SUPPORTED', 'ci_low': float(low), 'ci_high': float(high),
            'covers_truth': bool(low <= truth <= high), 'width': float(high - low),
            'lower_gt_0_505': bool(low > .505), 'lower_gt_0_47': bool(low > .47)}


def evaluate_sample(groups, rho, shift, replicate, *, draws=2000):
    y, probability, time, group, truth = sample(groups, rho, shift, replicate)
    approximate, _, _ = rank_influence(y, probability, time, group)
    point, boot, receipt = evaluate_partition(y, {'KNOWN_FIXED_SCORE': probability}, time, group,
        partition=f'CI_CALIBRATION_{groups}_{rho}_{shift}_{replicate}', draws=draws)
    estimate = point['KNOWN_FIXED_SCORE']['TS_AUC']
    assert estimate is not None and abs(estimate - approximate['TS_AUC']) < 1e-12
    values = boot['KNOWN_FIXED_SCORE']
    valid = np.isfinite(values)
    low, high = np.quantile(values[valid], [.025, .975]) if valid.sum() >= .95 * draws else (None, None)
    return {'groups': groups, 'rho': rho, 'mean_shift': shift, 'replicate': replicate,
            'rows': len(y), 'true_AUC': truth, 'estimated_AUC': estimate, 'bias': float(estimate - truth),
            'same_time_pairs': point['KNOWN_FIXED_SCORE']['pairs'],
            'methods': {'CLUSTER_NORMAL': interval_record(approximate['ci_low'], approximate['ci_high'], truth),
                        'EXACT_GROUP_PERCENTILE': interval_record(low, high, truth)},
            'cluster_SE': approximate['cluster_se'], 'bootstrap_valid_draws': int(valid.sum()),
            'bootstrap_draws': draws, 'resampling': receipt,
            'synthetic_values_sha256': hashlib.sha256(probability.tobytes()).hexdigest(),
            'synthetic_labels_sha256': hashlib.sha256(y.tobytes()).hexdigest(),
            'model_fit_count': 0, 'real_RP_values_or_predictions_used': False}


def wilson(successes, count):
    if not 0 <= successes <= count or count <= 0:
        raise ValueError('Finite nonempty binomial experiment required')
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    width = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    # At k=0/n these are exact mathematical endpoints; avoid cancellation
    # residuals being reported as evidence against zero/full coverage.
    return (0. if successes == 0 else max(0., center - width),
            1. if successes == count else min(1., center + width))


def aggregate(records):
    assert len(records) == 100 and {r['replicate'] for r in records} == set(range(100))
    first = records[0]
    assert all((r['groups'], r['rho'], r['mean_shift']) == (first['groups'], first['rho'], first['mean_shift']) for r in records)
    rows = []
    for method in ('CLUSTER_NORMAL', 'EXACT_GROUP_PERCENTILE'):
        valid = [r for r in records if r['methods'][method]['status'] == 'SUPPORTED']
        count = sum(r['methods'][method]['covers_truth'] for r in valid)
        lower, upper = wilson(count, len(valid)) if valid else (None, None)
        rows.append({'groups': first['groups'], 'rho': first['rho'], 'mean_shift': first['mean_shift'],
            'method': method, 'true_AUC': first['true_AUC'], 'replicates': 100, 'supported_intervals': len(valid),
            'coverage_count': count, 'coverage': count / len(valid) if valid else None,
            'coverage_MC95_low': lower, 'coverage_MC95_high': upper,
            'mean_bias': float(np.mean([r['bias'] for r in records])),
            'mean_interval_width': float(np.mean([r['methods'][method]['width'] for r in valid])) if valid else None,
            'lower_gt_0_505_count': sum(r['methods'][method]['lower_gt_0_505'] for r in valid),
            'lower_gt_0_47_count': sum(r['methods'][method]['lower_gt_0_47'] for r in valid),
            'minimum_finite_bootstrap_draws': min(r['bootstrap_valid_draws'] for r in records),
            'within_cell_MC_interval_not_simultaneous_across_cells': True,
            'fixed_score_sampling_only_not_selection_or_training_uncertainty': True})
    return rows
