"""Conservative finite-tree score envelope and age-map alarm impossibility bound."""
import math
import numpy as np


def probability_upper_bound(model):
    if hasattr(model,'primary') and hasattr(model,'complement'):
        left=probability_upper_bound(model.primary)['probability_upper_bound']
        right=probability_upper_bound(model.complement)['probability_upper_bound']
        value=np.float32(model.weight*left+(1.-model.weight)*right)
        upper=float(np.nextafter(value,np.float32(1.)))
        return {'probability_upper_bound':upper,'bound_type':'fixed probability blend of conservative component envelopes'}
    tree=model.compact
    margin=float(tree.bias)
    for k,start in enumerate(tree.roots):
        end=tree.roots[k+1] if k+1<len(tree.roots) else len(tree.values)
        leaves=tree.values[start:end][tree.features[start:end]<0]
        if not len(leaves) or not np.isfinite(leaves).all():
            raise ValueError('Every finite tree requires a leaf')
        margin+=float(leaves.max())
        margin=float(np.nextafter(margin,math.inf))
    # Upward guard exceeds accumulated floating summation/libm error here;
    # the final nextafter also rounds the float32 output bound outward.
    margin+=1e-10
    probability=1./(1.+math.exp(-margin))
    upper=float(np.nextafter(np.float32(probability),np.float32(1.)))
    return {'margin_upper_bound':margin,'probability_upper_bound':upper,
        'bound_type':'sum of each tree maximum leaf; simultaneous reachability not assumed'}


def mapped_upper(upper,time_online,beta):
    return upper/(upper+(1.-upper)*(1.+time_online/128.)**beta)


def no_alarm_after(upper,level,beta):
    """Sufficient online index beyond which q <= fixed level for every input."""
    if not 0<=upper<=1 or not 0<=level<=1 or beta not in (0.,.5,1.):
        raise ValueError('Valid probability, threshold and frozen beta required')
    if upper<=level:
        return 0
    if beta==0. or upper==1. or level==0.:
        return None
    ratio=upper*(1.-level)/(level*(1.-upper))
    root=128.*(ratio**(1./beta)-1.)
    if not math.isfinite(root) or root>1e15:
        return None
    index=max(0,math.ceil(root))
    # Guard floating computation at the strict-threshold boundary.
    while mapped_upper(upper,index,beta)>level:
        index+=1
    return index
