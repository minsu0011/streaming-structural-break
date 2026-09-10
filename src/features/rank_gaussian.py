"""Historical-only marginal Gaussianization before the existing causal engine."""
from pathlib import Path
import hashlib, json
import numpy as np
from numba import njit
from scipy.special import ndtri
from src.features.config import CONFIG
from src.features.historical_calibration import CalibrationPolicy, HistoricalCalibratedState, implementation_hash as calibration_hash

POLICY_PATH = Path(__file__).resolve().parents[2] / 'configs/rank_gaussian.json'
RANK_POLICY = json.loads(POLICY_PATH.read_text(encoding='utf-8'))


def implementation_hash(policy, config=CONFIG):
    return hashlib.sha256(Path(__file__).read_bytes() + json.dumps(RANK_POLICY, sort_keys=True).encode() + calibration_hash(policy, config).encode()).hexdigest()


@njit(cache=False)
def map_one(point, knots, scores):
    if not np.isfinite(point): return np.float32(point)
    j = np.searchsorted(knots, point, side='left')
    if j == 0: return np.float32(scores[0])
    if j == len(knots): return np.float32(scores[-1])
    fraction = (point - knots[j - 1]) / (knots[j] - knots[j - 1])
    return np.float32(scores[j - 1] + fraction * (scores[j] - scores[j - 1]))


@njit(cache=False)
def map_array(points, knots, scores):
    result = np.empty(len(points), dtype=np.float32)
    for i in range(len(points)): result[i] = map_one(float(points[i]), knots, scores)
    return result


def historical_mapping(historical):
    h = np.asarray(historical, dtype=np.float32)
    if h.ndim != 1: raise ValueError('Historical values must be one-dimensional')
    finite = h[np.isfinite(h)].astype(np.float64)
    if not len(finite): finite = np.array([0.])
    probabilities = np.linspace(RANK_POLICY['lower_probability'], RANK_POLICY['upper_probability'], RANK_POLICY['quantile_knots'])
    values = np.quantile(finite, probabilities)
    normal_scores = ndtri(probabilities)
    knots, inverse = np.unique(values, return_inverse=True)
    scores = np.bincount(inverse, weights=normal_scores) / np.bincount(inverse)
    if len(knots) == 1:
        center = knots[0]; scale = max(1., abs(center) * 1e-6)
        knots = np.array([center - scale, center, center + scale]); scores = np.array([-1., 0., 1.])
    return knots, scores, map_array(h, knots, scores)


class RankGaussianCalibratedState:
    def __init__(self, historical, *, config=CONFIG, policy=CalibrationPolicy(method='historical_median_mad')):
        self.knots, self.scores, transformed = historical_mapping(historical)
        self.inner = HistoricalCalibratedState(transformed, config=config, policy=policy)

    def update_and_get(self, point):
        transformed = map_one(float(point), self.knots, self.scores)
        return self.inner.update_and_get(transformed)

    @property
    def state_array_bytes(self):
        return self.knots.nbytes + self.scores.nbytes + self.inner.state_array_bytes
