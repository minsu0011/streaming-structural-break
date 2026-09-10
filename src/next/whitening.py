"""Stable historical-only AR innovation filters; O(p) streaming work.

No training labels, break position or online suffix enters this module.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
from numba import njit

AR_ORDERS = (1, 2, 4, 8, 12)
RIDGE_TRACE_FRACTION = .001
INPUT_CLIP = 12.
POLE_RADIUS_MAX = .995


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@njit(cache=True)
def _normal_equations(z, order):
    gram = np.zeros((order, order), dtype=np.float64)
    rhs = np.zeros(order, dtype=np.float64)
    for t in range(order, len(z)):
        for j in range(order):
            lag = z[t-1-j]
            rhs[j] += lag*z[t]
            for k in range(order):
                gram[j, k] += lag*z[t-1-k]
    return gram, rhs


@njit(cache=True)
def _innovations(z, coefficients):
    p = len(coefficients)
    result = np.empty(len(z)-p, dtype=np.float64)
    for t in range(p, len(z)):
        pred = 0.
        for j in range(p):
            pred += coefficients[j]*z[t-1-j]
        result[t-p] = z[t]-pred
    return result


@njit(cache=True)
def innovation_update(point, reference, coefficients, ring, counter):
    z = min(max((float(point)-reference[0])/reference[1], -INPUT_CLIP), INPUT_CLIP)-reference[2]
    pos = counter[0]
    prediction = 0.
    for j in range(len(coefficients)):
        prediction += coefficients[j]*ring[(pos-1-j) % len(ring)]
    ring[pos] = z
    counter[0] = (pos+1) % len(ring)
    return z-prediction


def fit_historical(historical, order=8):
    if order not in AR_ORDERS:
        raise ValueError('AR order outside preregistered family')
    h = np.asarray(historical, dtype=np.float32).astype(np.float64)
    if len(h) <= order+16 or not np.isfinite(h).all():
        raise ValueError('Finite historical segment with enough AR observations required')
    median = float(np.median(h))
    mad = float(1.4826*np.median(np.abs(h-median)))
    scale = max(mad, float(h.std())*.01, 1e-8)
    z = np.clip((h-median)/scale, -INPUT_CLIP, INPUT_CLIP)
    center = float(z.mean())
    z -= center
    gram, rhs = _normal_equations(z, order)
    ridge = max(float(np.trace(gram))/order*RIDGE_TRACE_FRACTION, 1e-8)
    coefficients = np.linalg.solve(gram+ridge*np.eye(order), rhs)
    roots = np.roots(np.r_[1., -coefficients])
    original_radius = float(np.max(np.abs(roots)))
    # Pole projection is deterministic and applied only from H fit coefficients.
    if original_radius > POLE_RADIUS_MAX:
        roots *= np.minimum(1., POLE_RADIUS_MAX/np.maximum(np.abs(roots), 1e-15))
        coefficients = -np.poly(roots).real[1:]
    residual = _innovations(z, coefficients)
    ring = z[-order:].copy()
    return {'reference': np.array([median, scale, center]), 'coefficients': coefficients,
            'ring': ring, 'counter': np.zeros(1, dtype=np.int64), 'historical_innovations': residual,
            'original_pole_radius': original_radius, 'pole_projected': original_radius > POLE_RADIUS_MAX}


@dataclass
class InnovationState:
    fitted: dict

    @classmethod
    def from_historical(cls, historical, order=8):
        return cls(fit_historical(historical, order))

    def update(self, point):
        f = self.fitted
        return innovation_update(point, f['reference'], f['coefficients'], f['ring'], f['counter'])
