"""Historical AR(4) innovation input, with constant-memory online residuals."""
from pathlib import Path
import hashlib, json
import numpy as np
from numba import njit
from src.features.config import CONFIG
from src.features.streaming import StreamingFeatureState
from src.features.historical_calibration import CalibrationPolicy, HistoricalCalibratedState, implementation_hash as calibration_hash

POLICY_PATH = Path(__file__).resolve().parents[2] / 'configs/ar_residual_input.json'
AR_POLICY = json.loads(POLICY_PATH.read_text(encoding='utf-8'))


@njit(cache=False)
def historical_normal_equations(values, order):
    gram=np.zeros((order,order),dtype=np.float64);rhs=np.zeros(order,dtype=np.float64)
    for t in range(order,len(values)):
        for j in range(order):
            previous=values[t-1-j];rhs[j]+=previous*values[t]
            for k in range(order):gram[j,k]+=previous*values[t-1-k]
    return gram,rhs


@njit(cache=False)
def historical_innovations(values, coefficients):
    order=len(coefficients);result=np.empty(len(values)-order,dtype=np.float32)
    for t in range(order,len(values)):
        prediction=0.
        for j in range(order):prediction+=coefficients[j]*values[t-1-j]
        result[t-order]=values[t]-prediction
    return result


def implementation_hash(policy, config=CONFIG):
    return hashlib.sha256(Path(__file__).read_bytes() + json.dumps(AR_POLICY, sort_keys=True).encode() + calibration_hash(policy, config).encode()).hexdigest()


def historical_filter(historical, config=CONFIG):
    h = np.asarray(historical, dtype=np.float32)
    reference = StreamingFeatureState(h, config=config).reference
    location, scale = reference[:2]
    z = np.clip((np.where(np.isfinite(h), h, location).astype(np.float64) - location) / scale, -config.clip_z, config.clip_z)
    order = AR_POLICY['order']; coefficients = np.zeros(order, dtype=np.float64)
    if len(z) > order:
        gram,rhs=historical_normal_equations(z,order)
        ridge = max(float(np.trace(gram)) / order * AR_POLICY['ridge_trace_fraction'], config.moment_floor)
        coefficients = np.linalg.solve(gram + ridge * np.eye(order), rhs)
        norm = float(np.linalg.norm(coefficients))
        if norm > AR_POLICY['coefficient_l2_limit']: coefficients *= AR_POLICY['coefficient_l2_limit'] / norm
        residual = historical_innovations(z,coefficients)
    else: residual = z.astype(np.float32)
    ring = np.zeros(order, dtype=np.float64)
    if len(z): ring[-min(order, len(z)):] = z[-order:]
    return np.array([location, scale, config.clip_z]), coefficients, ring, residual


@njit(cache=False)
def innovation_one(point, reference, coefficients, ring, counter):
    missing = not np.isfinite(point)
    z = 0.0 if missing else min(max((point - reference[0]) / reference[1], -reference[2]), reference[2])
    position = counter[0]; prediction = 0.0
    for j in range(len(coefficients)): prediction += coefficients[j] * ring[(position - 1 - j) % len(ring)]
    ring[position] = z; counter[0] = (position + 1) % len(ring)
    return np.float32(np.nan if missing else z - prediction)


class ARResidualCalibratedState:
    def __init__(self, historical, *, config=CONFIG, policy=CalibrationPolicy(method='historical_median_mad')):
        self.reference, self.coefficients, self.ring, residual = historical_filter(historical, config)
        self.counter = np.zeros(1, dtype=np.int64)
        self.inner = HistoricalCalibratedState(residual, config=config, policy=policy)

    def update_and_get(self, point):
        residual = innovation_one(float(point), self.reference, self.coefficients, self.ring, self.counter)
        return self.inner.update_and_get(residual)

    @property
    def state_array_bytes(self):
        return sum(x.nbytes for x in (self.reference, self.coefficients, self.ring, self.counter)) + self.inner.state_array_bytes
