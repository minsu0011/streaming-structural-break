"""One preregistered fixed integration, with a complete-procedure inner gate."""
import numpy as np


def fixed_blend(baseline,model):
    b=np.asarray(baseline,dtype=np.float64);m=np.asarray(model,dtype=np.float64)
    if b.shape!=m.shape or not np.isfinite(b).all() or not np.isfinite(m).all():raise ValueError('Exact finite matched integration inputs')
    if np.any((b<0)|(b>1)|(m<0)|(m>1)):raise ValueError('Probability integration only')
    return .95*b+.05*m


def integration_eligible(comparison):
    return bool(comparison.get('status')=='SUPPORTED' and len(comparison['fold_deltas'])==3 and
                comparison['mean_delta']>=.0001 and comparison['positive_folds']>=2 and comparison['ci_low']>0)
