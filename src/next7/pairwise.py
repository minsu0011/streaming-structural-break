"""G3: fixed within-time pairwise logistic correction, same G2 feature capacity."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits
from src.next7.model import Correction, logit, implementation_hash as correction_hash
from src.next7.features import FULL


def implementation_hash():
    return hashlib.sha256(Path(__file__).read_bytes() + correction_hash().encode()).hexdigest()


@dataclass
class PairwiseCorrection(Correction):
    pairwise_source_sha256: str

    def validate(self):
        super().validate()
        if self.pairwise_source_sha256 != implementation_hash():
            raise RuntimeError('Fitted G3 pairwise implementation changed')


def make_pair_plan(target, time_online, series_for_audit, *, cap=1024, seed=20260908):
    y, t, series = np.asarray(target), np.asarray(time_online), np.asarray(series_for_audit)
    if y.ndim != 1 or t.shape != y.shape or series.shape != y.shape or len(y) == 0:
        raise ValueError('Aligned target/time/audit-group arrays required')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(t).all() or np.any(t < 0) or np.any(t != np.floor(t)):
        raise ValueError('Binary targets and nonnegative integer times required')
    if not isinstance(cap, int) or cap <= 0:
        raise ValueError('Positive pair cap required')
    order = np.argsort(t, kind='stable')
    boundaries = np.r_[0, np.flatnonzero(np.diff(t[order])) + 1, len(order)]
    rng = np.random.default_rng(seed)
    positive, negative, weights, times, records = [], [], [], [], []
    total_mass = 0
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        rows = order[start:end]
        if len(np.unique(series[rows])) != len(rows):
            raise ValueError('Each history must have one observation per online time')
        pos, neg = rows[y[rows] == 1], rows[y[rows] == 0]
        mass = len(pos) * len(neg)
        if mass == 0:
            continue
        total_mass += mass
        count = min(mass, cap)
        if mass <= cap:
            p, n = np.repeat(pos, len(neg)), np.tile(neg, len(pos))
        else:
            p = pos[rng.integers(0, len(pos), size=count)]
            n = neg[rng.integers(0, len(neg), size=count)]
        positive.append(p)
        negative.append(n)
        weights.append(np.full(count, mass / count))
        times.append(np.full(count, int(t[rows[0]]), dtype=np.int64))
        records.append({'time': int(t[rows[0]]), 'positive_rows': len(pos), 'negative_rows': len(neg),
            'eligible_pair_mass': mass, 'sampled_pairs': count, 'exhaustive': mass <= cap})
    if not total_mass:
        raise ValueError('No eligible within-time positive-negative pairs')
    p, n = np.concatenate(positive), np.concatenate(negative)
    w, at = np.concatenate(weights) / total_mass, np.concatenate(times)
    assert np.all(t[p] == t[n]) and np.all(y[p] == 1) and np.all(y[n] == 0)
    assert np.all(series[p] != series[n])
    assert abs(w.sum() - 1.) < 1e-12
    return {'positive_row': p, 'negative_row': n, 'weights': w, 'time': at,
        'time_records': records, 'total_pair_mass': total_mass, 'cap': cap, 'seed': seed}


def plan_hash(plan):
    h = hashlib.sha256()
    for key in ('positive_row', 'negative_row', 'weights', 'time'):
        array = np.ascontiguousarray(plan[key])
        h.update(key.encode())
        h.update(array.dtype.str.encode())
        h.update(array.tobytes())
    return h.hexdigest()


def objective(beta, differences, offset_differences, weights, penalty=.01):
    margin = offset_differences + differences @ beta
    loss = float(weights @ np.logaddexp(0., -margin) + .5 * penalty * (beta @ beta))
    gradient = differences.T @ (-weights * expit(-margin)) + penalty * beta
    return loss, gradient


def fit_pairwise(features, baseline, target, time_online, series_for_audit, provenance, *, frozen_plan=None):
    if not provenance.get('strict_nested_OOF') or not provenance.get('outer_validation_excluded'):
        raise RuntimeError('LEAKAGE_FAIL: strict nested provenance required')
    x, b, y = np.asarray(features, dtype=np.float64), np.asarray(baseline, dtype=np.float64), np.asarray(target)
    if x.ndim != 2 or x.shape[1] != 10 or b.shape != (len(x),) or y.shape != b.shape:
        raise ValueError('Exactly ten aligned G2 features and baseline/targets required')
    if not np.isfinite(x).all() or not np.isfinite(b).all():
        raise ValueError('Finite features and baseline required')
    # Match the frozen parent's advanced-column selection AND reduction layout.
    # Reducing a C-contiguous matrix directly changes the final rounding bits.
    x = x[:, FULL]
    plan = make_pair_plan(y, time_online, series_for_audit)
    if frozen_plan is not None and plan_hash(plan) != plan_hash(frozen_plan):
        raise RuntimeError('Frozen pair plan changed')
    center = x.mean(axis=0)
    spread = np.maximum(x.std(axis=0), 1e-4)
    z = np.clip((x - center) / spread, -8., 8.)
    offset = logit(b)
    p, n, w = plan['positive_row'], plan['negative_row'], plan['weights']
    differences = np.ascontiguousarray(z[p] - z[n])
    offsets = offset[p] - offset[n]
    with threadpool_limits(limits=1):
        initial, initial_gradient = objective(np.zeros(10), differences, offsets, w)
        result = minimize(objective, np.zeros(10), args=(differences, offsets, w), method='L-BFGS-B', jac=True,
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
    if not result.success:
        raise RuntimeError('Fixed G3 optimizer did not converge: ' + str(result.message))
    training = {**provenance, 'optimizer_success': bool(result.success), 'iterations': int(result.nit),
        'objective': float(result.fun), 'initial_objective': initial,
        'initial_gradient': initial_gradient.tolist(), 'gradient_max_abs': float(np.max(np.abs(result.jac))),
        'penalty': .01, 'intercept': False, 'baseline_offset_fixed_coefficient': 1,
        'pair_plan_sha256': plan_hash(plan), 'sampled_pairs': len(p), 'eligible_pair_mass': plan['total_pair_mass'],
        'sampling_seed': 20260908, 'per_time_cap': 1024, 'pair_time_records': plan['time_records'],
        'objective_name': 'within-time pairwise logistic', 'groups_are_audit_metadata_not_features': True}
    return PairwiseCorrection('G3_PAIRWISE_FULL', FULL, center, spread, result.x, None, training,
        correction_hash(), implementation_hash())
