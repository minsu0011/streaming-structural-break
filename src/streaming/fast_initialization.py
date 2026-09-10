"""Prediction-preserving initialization: share one historical base-state setup."""
from dataclasses import replace
import hashlib
from pathlib import Path
import numpy as np
from src.features.streaming import StreamingFeatureState, FEATURE_NAMES, FEATURE_FAMILIES
from src.features.historical_calibration import _historical_replay
from src.streaming.fused import PreparedFusedPredictor, FusedTreeDetector, implementation_hash as fused_hash


def implementation_hash():
    return hashlib.sha256(Path(__file__).read_bytes() + fused_hash().encode()).hexdigest()


def shared_mad_reference(historical, state, policy, selected_columns=None):
    """Preserve online state and the exact calibration of every required column.

    Without selected_columns all calibration arrays retain the original contract.
    With fitted tree columns, unused output slots remain uncalibrated; no tree
    split can inspect them. An empty intersection needs no historical replay.
    """
    if policy.method != 'historical_median_mad': raise ValueError('Exact optimization is limited to historical median/MAD')
    mask = np.array([family in policy.signed_families for family in FEATURE_FAMILIES]); mask[[3, 4, 5]] = False
    chosen = None
    if selected_columns is not None:
        chosen = np.intersect1d(np.flatnonzero(mask), np.asarray(selected_columns, dtype=np.int64))
        mask = np.zeros(len(FEATURE_NAMES), dtype=np.bool_); mask[chosen] = True
        center = np.zeros(len(FEATURE_NAMES), dtype=np.float64); scale = np.ones(len(FEATURE_NAMES), dtype=np.float64)
        if not len(chosen): return center, scale, mask
    h = np.asarray(historical, dtype=np.float32)
    reference = _historical_replay(h, state.reference, state.baseline, state.thresholds,
        state.ring.copy(), state.ew.copy(), state.counters.copy(), state.output.copy(), state.parameters, state.alphas, state.lags)
    if not len(reference): reference = np.zeros((1, len(FEATURE_NAMES)), dtype=np.float32)
    burn = min(policy.burn_in, max(0, len(reference) // 2))
    # Partition contiguous feature rows. Values and midpoint arithmetic are
    # identical to the original axis-0 median/MAD calculation.
    columns = np.ascontiguousarray((reference[burn:] if chosen is None else reference[burn:, chosen]).T, dtype=np.float64)
    location = np.median(columns, axis=1)
    deviation = np.maximum(np.median(np.abs(columns - location[:, None]), axis=1) * 1.4826, policy.standard_deviation_floor)
    if chosen is None: center, scale = location, deviation
    else: center[chosen], scale[chosen] = location, deviation
    return center, scale, mask


class PreparedFastInitFusedPredictor(PreparedFusedPredictor):
    def new_state(self, historical): return FastInitFusedTreeDetector(self, historical)


class FastInitFusedTreeDetector(FusedTreeDetector):
    def __init__(self, prepared, historical):
        bundle = prepared.bundle
        method = bundle.policy.method.removesuffix('_startup') if prepared.calibrated else None
        if not prepared.calibrated or method != 'historical_median_mad':
            super().__init__(prepared, historical); return
        self.prepared = prepared
        self.state = StreamingFeatureState(historical, config=bundle.feature_config)
        policy = replace(bundle.policy, method=method)
        center, scale, mask = shared_mad_reference(historical, self.state, policy, selected_columns=np.unique(prepared.tree_args[1]))
        output = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        corrected = np.zeros(len(FEATURE_NAMES) if prepared.startup else 1, dtype=np.float32)
        self.calibration_args = (center, scale, mask, float(bundle.policy.clip), output, corrected)
        s = self.state
        self.state_args = (s.reference, s.baseline, s.thresholds, s.ring, s.ew, s.counters, s.output, s.parameters, s.alphas, s.lags)
