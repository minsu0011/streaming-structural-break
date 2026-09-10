"""One compiled call for AR innovation, calibrated features and scalar trees."""
import hashlib
from pathlib import Path
import numpy as np
from numba import njit
from src.features.ar_residual_input import historical_filter, innovation_one
from src.features.streaming import StreamingFeatureState, FEATURE_NAMES
from src.features.historical_calibration import historical_reference
from src.streaming.fused import PreparedFusedPredictor, fused_predict
from src.streaming.fast_initialization import shared_mad_reference, implementation_hash as initialization_hash


def implementation_hash():
    sources = [Path(__file__), Path(__file__).parents[1] / 'features/ar_residual_input.py', Path(__file__).parents[1] / 'features/ar_order_input.py']
    return hashlib.sha256(b''.join(path.read_bytes() for path in sources) + initialization_hash().encode()).hexdigest()


@njit(cache=False)
def fused_ar_predict(point, ar_args, state_args, calibration_args, tree_args, is_xgboost):
    residual = innovation_one(point, *ar_args)
    return fused_predict(float(residual), state_args, calibration_args, tree_args, True, is_xgboost, False)


class PreparedFusedARPredictor(PreparedFusedPredictor):
    def __init__(self, bundle, fast_initialization=False):
        from src.models.ar_calibrated import ARCalibratedBundle
        from src.models.ar_order_calibrated import AROrderCalibratedBundle
        if not isinstance(bundle, (ARCalibratedBundle, AROrderCalibratedBundle)): raise ValueError('AR fused inference requires an explicit AR bundle')
        super().__init__(bundle)
        self.fast_initialization = fast_initialization; self.source_hash = implementation_hash()
    def new_state(self, historical): return FusedARDetector(self, historical)


class FusedARDetector:
    def __init__(self, prepared, historical):
        self.prepared = prepared; bundle = prepared.bundle
        if prepared.fast_initialization:
            if hasattr(bundle, 'input_order'):
                from src.features.ar_order_input import historical_filter as order_filter
                reference, coefficients, ring, residual = order_filter(historical, bundle.feature_config, bundle.input_order)
            else: reference, coefficients, ring, residual = historical_filter(historical, bundle.feature_config)
            self.ar_args = (reference, coefficients, ring, np.zeros(1, dtype=np.int64))
            self.state = StreamingFeatureState(residual, config=bundle.feature_config)
            if bundle.policy.method == 'historical_median_mad': center, scale, mask = shared_mad_reference(residual, self.state, bundle.policy, selected_columns=np.unique(prepared.tree_args[1]))
            else: center, scale, mask = historical_reference(residual, config=bundle.feature_config, policy=bundle.policy)
            output = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        else:
            ar_state = bundle.make_state(historical); self.ar_args = (ar_state.reference, ar_state.coefficients, ar_state.ring, ar_state.counter)
            inner = ar_state.inner; self.state = inner.state; center, scale, mask, output = inner.center, inner.scale, inner.mask, inner.output
        self.calibration_args = (center, scale, mask, float(bundle.policy.clip), output, np.zeros(1, dtype=np.float32))
        s = self.state
        self.state_args = (s.reference, s.baseline, s.thresholds, s.ring, s.ew, s.counters, s.output, s.parameters, s.alphas, s.lags)

    def predict_one(self, point):
        return float(fused_ar_predict(float(point), self.ar_args, self.state_args, self.calibration_args, self.prepared.tree_args, self.prepared.is_xgboost))

    @property
    def state_array_bytes(self):
        return self.state.state_array_bytes + sum(x.nbytes for x in self.ar_args) + sum(x.nbytes for x in self.calibration_args if isinstance(x, np.ndarray))
