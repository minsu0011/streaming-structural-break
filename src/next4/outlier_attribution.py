"""Algebraic allocation of paired Tree-SHAP margin changes; no causal claim."""
import numpy as np
from scipy.special import expit


def group_name(candidate,name):
    if candidate.startswith(('B1_','B2_')):
        if name.startswith('arch8__'):return 'old_ARCH'
        if name.startswith('serial_variance__raw_energy_'):return 'old_serial_raw_energy'
        if name.startswith('serial_variance__log_variance_forecast_'):return 'old_serial_log_forecast'
        for channel in ('raw_energy','conditional_energy'):
            for direction in ('lower','upper'):
                if '_'+channel+'_'+direction+'_' in name:return 'new_'+channel+'_'+direction
    elif candidate.startswith('D2_'):
        if name.endswith('_upper_minus_lower'):return 'window_direction_gap'
        if name.endswith('_lower_log_mixture'):return 'lower_scale_mixture'
        if name.endswith('_upper_log_mixture'):return 'upper_scale_mixture'
        if name.endswith('_log_mixture'):return 'two_sided_window_mixture'
        if name.startswith('next4_scale_mixture_'):return 'pooled_age_evidence'
    elif candidate.startswith('D3_'):
        for horizon in (32,128,512):
            if name.startswith('next4_hazard_'+str(horizon)+'_'):return 'hazard_'+str(horizon)
        if name in ('next4_short_minus_long_hazard','next4_medium_minus_long_hazard'):return 'between_hazard_contrasts'
    raise ValueError('Feature outside the frozen attribution grouping: '+candidate+' '+name)


def allocate_probability_change(clean_shap,altered_shap,*,expert_weight):
    a=np.asarray(clean_shap,dtype=np.float64);b=np.asarray(altered_shap,dtype=np.float64)
    if a.ndim!=2 or a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or not 0<expert_weight<=1:
        raise ValueError('Aligned finite binary raw-margin contributions required')
    ma=a.sum(axis=1);mb=b.sum(axis=1);pa=expit(ma);pb=expit(mb);dm=mb-ma
    midpoint=expit((ma+mb)/2);slope=midpoint*(1-midpoint)
    np.divide(pb-pa,dm,out=slope,where=np.abs(dm)>1e-10)
    allocated=expert_weight*(b-a)*slope[:,None]
    probability_delta=expert_weight*(pb-pa)
    return allocated,probability_delta
