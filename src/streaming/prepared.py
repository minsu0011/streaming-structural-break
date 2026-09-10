"""Dispatch production kernels and their complete source identity receipts."""
from src.models.ar_calibrated import ARCalibratedBundle
from src.models.ar_order_calibrated import AROrderCalibratedBundle
from src.models.cross_order import CrossOrderBlendBundle
from src.models.arch_calibrated import ARCHCalibratedBundle


def prepare_model(model,*,fast_initialization=False):
    if isinstance(model,ARCHCalibratedBundle):
        from src.streaming.fused_arch import PreparedFusedARCHPredictor
        return PreparedFusedARCHPredictor(model,fast_initialization=fast_initialization)
    if isinstance(model,CrossOrderBlendBundle):
        from src.streaming.fused_cross_order import PreparedFusedCrossOrderPredictor
        return PreparedFusedCrossOrderPredictor(model,fast_initialization=fast_initialization)
    if isinstance(model,(ARCalibratedBundle,AROrderCalibratedBundle)):
        from src.streaming.fused_ar import PreparedFusedARPredictor
        return PreparedFusedARPredictor(model,fast_initialization=fast_initialization)
    if fast_initialization:
        from src.streaming.fast_initialization import PreparedFastInitFusedPredictor
        return PreparedFastInitFusedPredictor(model)
    from src.streaming.fused import PreparedFusedPredictor
    return PreparedFusedPredictor(model)


def source_hashes(model,*,fast_initialization=False):
    from src.streaming.fused import implementation_hash as fused_hash
    from src.streaming.fast_initialization import implementation_hash as init_hash
    from src.streaming.fused_ar import implementation_hash as ar_hash
    from src.streaming.fused_cross_order import implementation_hash as cross_hash
    from src.streaming.fused_arch import implementation_hash as arch_hash
    is_ar=isinstance(model,(ARCalibratedBundle,AROrderCalibratedBundle,CrossOrderBlendBundle,ARCHCalibratedBundle))
    return {'fused_source_hash':fused_hash(),'initialization_source_hash':init_hash() if fast_initialization else None,
        'ar_fused_source_hash':ar_hash() if is_ar else None,'cross_order_fused_source_hash':cross_hash() if isinstance(model,CrossOrderBlendBundle) else None,
        'arch_fused_source_hash':arch_hash() if isinstance(model,ARCHCalibratedBundle) else None}
