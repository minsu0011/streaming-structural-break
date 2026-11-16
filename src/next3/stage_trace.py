"""Read-only diagnostics along the exact frozen variance/dependence pipeline."""
import numpy as np
from numba import njit
from src.next.channel_evidence import CHANNEL_STATS,_logadd
from src.next2.shared_blend_runtime import shared_feature_step
from src.models.compact_trees import tree_margin

CHANNELS=('raw_energy','log_variance_forecast')
NAMES=tuple(channel+'_'+stat for channel in CHANNELS for stat in CHANNEL_STATS)+(
    'raw_energy_before_H_standardization','forecast_before_H_standardization',
    'raw_energy_after_H_before_clip','forecast_after_H_before_clip')+tuple(
    channel+'_'+direction+'_bayes'+prior for channel in CHANNELS
    for direction in ('negative','positive') for prior in ('025','1'))+('J102_score','C205_score','BL80_score')


@njit(cache=False)
def directional_bayes(args,variances):
    ages,widths,ewma,cusum,cumulative,prefix,counter,memory,output=args
    t=counter[0]
    result=np.empty(8,dtype=np.float64)
    for channel in range(2):
        bins=np.full(4,-1e300)
        width=0.
        for k in range(len(ages)):
            age=ages[k]
            if age>t:break
            width+=widths[k]
            total=cumulative[channel]-prefix[(t-age)%len(prefix),channel]
            ratio=age/variances[channel,k]
            offset=0 if total<0 else 2
            for j,prior in enumerate((.25,1.)):
                term=np.log(widths[k])-.5*np.log1p(prior*age*ratio)+.5*prior*ratio*ratio*total*total/(1.+prior*age*ratio)
                bins[offset+j]=_logadd(bins[offset+j],term)
        for k in range(4):result[channel*4+k]=bins[k]-np.log(width)
    return result


@njit(cache=False)
def replay_stage_trace(points,args):
    result=np.empty((len(points),len(NAMES)),dtype=np.float64)
    for t in range(len(points)):
        feature=shared_feature_step(float(points[t]),args)
        shared=args[1]
        pair=shared[5][-1]
        for k in range(26):result[t,k]=pair[k]
        result[t,26]=shared[4][0]*shared[3][1,0]+shared[3][0,0]
        result[t,27]=shared[4][1]*shared[3][1,1]+shared[3][0,1]
        result[t,28]=shared[4][0]
        result[t,29]=shared[4][1]
        values=directional_bayes(shared[5],shared[6])
        for k in range(8):result[t,30+k]=max(values[k],-80.)
        a_tree,b_tree,weight=args[-1]
        a=np.float32(1./(1.+np.exp(-tree_margin(feature,*a_tree))))
        b=np.float32(1./(1.+np.exp(-tree_margin(shared[-1][-1],*b_tree))))
        result[t,38]=a;result[t,39]=b
        result[t,40]=np.float32(weight*np.float64(a)+(1.-weight)*np.float64(b))
    return result


def orientation_for_decrease(name):
    if name in ('J102_score','C205_score','BL80_score'):return 1.
    if 'positive_bayes' in name:return -1.
    if name.endswith(('_current','_ewma8','_ewma32','_ewma128','_before_H_standardization','_after_H_before_clip')):return -1.
    if name.endswith('_cusum_positive'):return -1.
    return 1.
