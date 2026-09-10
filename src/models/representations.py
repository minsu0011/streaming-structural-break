"""Explicit dispatch for fitted feature representations and their live contracts."""
from src.features.config import CONFIG
from src.features.streaming import StreamingFeatureState
from src.features.historical_calibration import CalibrationPolicy, HistoricalCalibratedState
from src.models.learners import ModelBundle
from src.models.calibrated import CalibratedBundle

AR_PREFIX = 'ar_residual_input_'
AR_ORDER_PREFIXES = ('ar_order2_', 'ar_order8_')
ARCH_PREFIX='ar8_arch1_'


def ar_input_order(representation):
    for order, prefix in zip((2, 8), AR_ORDER_PREFIXES):
        if representation.startswith(prefix): return order
    return None


def ar_prefix(representation):
    return next((prefix for prefix in (AR_PREFIX, *AR_ORDER_PREFIXES,ARCH_PREFIX) if representation.startswith(prefix)), '')


def calibration_policy(representation):
    prefix = ar_prefix(representation)
    method = representation.removeprefix(prefix)
    clip=24.0 if method.endswith('_clip24') else 12.0
    if clip==24.0:
        if not prefix:raise ValueError('The prospective wider-clip family is restricted to historical AR representations')
        method=method.removesuffix('_clip24')
    supported = ('historical_mean_std', 'historical_median_mad')
    if not prefix: supported += ('historical_mean_std_startup', 'historical_median_mad_startup')
    if method not in supported: raise ValueError('Unsupported fitted feature representation')
    return CalibrationPolicy(method=method,clip=clip)


def make_state(historical, config, representation):
    if representation=='cross_order_ar4_ar8_equal':
        from src.features.cross_order import CrossOrderFeatureState
        return CrossOrderFeatureState(historical,config=config)
    if representation == 'base': return StreamingFeatureState(historical, config=config)
    policy = calibration_policy(representation)
    if representation.startswith(ARCH_PREFIX):
        from src.features.arch_input import ARCHCalibratedState
        return ARCHCalibratedState(historical,config=config,policy=policy)
    if ar_input_order(representation) is not None:
        from src.features.ar_order_input import AROrderCalibratedState
        return AROrderCalibratedState(historical, config=config, policy=policy, input_order=ar_input_order(representation))
    if representation.startswith(AR_PREFIX):
        from src.features.ar_residual_input import ARResidualCalibratedState
        return ARResidualCalibratedState(historical, config=config, policy=policy)
    if policy.method.endswith('_startup'):
        from src.features.startup_calibration import StartupCalibratedState
        return StartupCalibratedState(historical, config=config, policy=policy)
    return HistoricalCalibratedState(historical, config=config, policy=policy)


def bundle_class(representation):
    if representation=='cross_order_ar4_ar8_equal':
        from src.models.cross_order import CrossOrderBlendBundle
        return CrossOrderBlendBundle
    if representation == 'base': return ModelBundle
    calibration_policy(representation)
    if representation.startswith(ARCH_PREFIX):
        from src.models.arch_calibrated import ARCHCalibratedBundle
        return ARCHCalibratedBundle
    if ar_input_order(representation) is not None:
        from src.models.ar_order_calibrated import AROrderCalibratedBundle
        return AROrderCalibratedBundle
    if representation.startswith(AR_PREFIX):
        from src.models.ar_calibrated import ARCalibratedBundle
        return ARCalibratedBundle
    return CalibratedBundle


def wrap_model(model, representation, config=CONFIG):
    if representation=='cross_order_ar4_ar8_equal':
        if not isinstance(model,bundle_class(representation)):raise ValueError('Cross-order fitting requires its explicit two-child learner')
        return model
    if representation == 'base': return model
    if ar_input_order(representation) is not None:
        return bundle_class(representation)(model, calibration_policy(representation), config, input_order=ar_input_order(representation))
    return bundle_class(representation)(model, calibration_policy(representation), config)


def deployment_kind(representation):
    if representation=='cross_order_ar4_ar8_equal':return 'cross_order_equal'
    if representation.startswith(ARCH_PREFIX):
        calibration_policy(representation)
        return 'ar8_arch1_calibrated'
    if representation == 'base': return 'supervised'
    calibration_policy(representation)
    if ar_input_order(representation) is not None: return 'ar_order_calibrated'
    return 'ar_residual_calibrated' if representation.startswith(AR_PREFIX) else 'historical_calibrated'


def representation_feature_names(representation,config=CONFIG):
    if representation=='cross_order_ar4_ar8_equal':
        from src.features.cross_order import feature_names
    else:
        from src.features.streaming import feature_names
    return feature_names(config)
