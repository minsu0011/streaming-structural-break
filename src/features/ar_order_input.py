"""Explicit AR-order refinements, preserving the original AR(4) source identity."""
from pathlib import Path
import hashlib
import numpy as np
from src.features.config import CONFIG
from src.features.streaming import StreamingFeatureState
from src.features.historical_calibration import CalibrationPolicy, HistoricalCalibratedState
from src.features.ar_residual_input import AR_POLICY, ARResidualCalibratedState, historical_normal_equations, historical_innovations, implementation_hash as legacy_hash


def implementation_hash(policy, config=CONFIG, input_order=4):
    return hashlib.sha256(Path(__file__).read_bytes() + legacy_hash(policy, config).encode() + str(input_order).encode()).hexdigest()


def historical_filter(historical, config=CONFIG, input_order=4):
    if input_order not in (2, 4, 8): raise ValueError('AR order is outside the declared small family')
    h = np.asarray(historical, dtype=np.float32)
    reference = StreamingFeatureState(h, config=config).reference
    location, scale = reference[:2]
    z = np.clip((np.where(np.isfinite(h), h, location).astype(np.float64) - location) / scale, -config.clip_z, config.clip_z)
    order = input_order; coefficients = np.zeros(order, dtype=np.float64)
    if len(z) > order:
        gram, rhs = historical_normal_equations(z, order)
        ridge = max(float(np.trace(gram)) / order * AR_POLICY['ridge_trace_fraction'], config.moment_floor)
        coefficients = np.linalg.solve(gram + ridge * np.eye(order), rhs)
        norm = float(np.linalg.norm(coefficients))
        if norm > AR_POLICY['coefficient_l2_limit']: coefficients *= AR_POLICY['coefficient_l2_limit'] / norm
        residual = historical_innovations(z, coefficients)
    else: residual = z.astype(np.float32)
    ring = np.zeros(order, dtype=np.float64)
    if len(z): ring[-min(order, len(z)):] = z[-order:]
    return np.array([location, scale, config.clip_z]), coefficients, ring, residual


class AROrderCalibratedState(ARResidualCalibratedState):
    def __init__(self, historical, *, config=CONFIG, policy=CalibrationPolicy(method='historical_median_mad'), input_order=4):
        self.reference, self.coefficients, self.ring, residual = historical_filter(historical, config, input_order)
        self.counter = np.zeros(1, dtype=np.int64)
        self.inner = HistoricalCalibratedState(residual, config=config, policy=policy)
