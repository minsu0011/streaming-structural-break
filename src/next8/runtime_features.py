"""Execution-only pruning of the frozen same-reference feature engine."""
import numpy as np
from numba import njit
from src.next4.scale_asymmetry import OLD_COLUMNS,appended_step,branch_values
from src.next.fused_runtime import _arch_feature_step
from src.next.serial_variance_evidence import serial_variance_step
from src.next.whitening import innovation_update
from src.next.variance_evidence import variance_channel_update
from src.next2.dependence_runtime import step as dependence_step
from src.next4.sequential import step as sequential_step


@njit(cache=True)
def branch_step(point,branch_args,output):
    # Exact branch portion of frozen scale_asymmetry.appended_step. It has
    # independent AR/ARCH buffers, so unused old/serial branches can be skipped.
    filter_args,parameters,conditional,mode,reference,kind,raw,work,ewma,cusum,counter=branch_args
    residual=innovation_update(point,*filter_args)
    variance_channel_update(residual,parameters,conditional,mode,raw)
    branch_values(raw,reference,kind,work);t=counter[0]+1
    for channel in range(4):
        j=channel//2;direction=channel%2
        value=(work[channel]-reference[j,3+direction])/reference[j,5+direction]
        k=77+channel*4;output[k]=work[channel]
        for h,half in enumerate((8.,32.)):
            alpha=1.-np.exp(-np.log(2.)/half)
            ewma[channel,h]=(1.-alpha)*ewma[channel,h]+alpha*value
            normalizer=np.sqrt(max(alpha/(2.-alpha)*(1.-(1.-alpha)**(2*t)),1e-8))
            output[k+1+h]=ewma[channel,h]/normalizer
        cusum[channel]=max(0.,cusum[channel]+min(max(value,-12.),12.)-.25)
        output[k+3]=np.log1p(cusum[channel])
    counter[0]=t


@njit(cache=True)
def raw_step(point,args):
    b,c,d,flags,raw=args
    if flags[0] and flags[1] and flags[2]:
        values=appended_step(point,*b)
        for j in range(93):raw[j]=values[j]
    else:
        if flags[0]:
            old=_arch_feature_step(point,b[0])
            for j in range(len(OLD_COLUMNS)):raw[j]=old[OLD_COLUMNS[j]]
        if flags[1]:
            serial=serial_variance_step(point,*b[1])
            for j in range(26):raw[51+j]=serial[j]
        if flags[2]:
            branch_step(point,b[2],b[3])
            for j in range(77,93):raw[j]=b[3][j]
    if flags[3]:
        values=dependence_step(point,*c)
        for j in range(24):raw[93+j]=values[j]
    if flags[4]:
        values=sequential_step(point,*d)
        for j in range(36):raw[117+j]=values[j]
    return raw


@njit(cache=True)
def normalized_step(point,engine,normalization):
    center,spread,used,output=normalization
    raw=raw_step(point,engine)
    for j in used:
        value=(np.float64(raw[j])-center[j])/spread[j]
        output[j]=np.float32(min(max(value,-12.),12.))
    return output


def raw_arguments(feature_state,used):
    if not feature_state.engines:raise ValueError('Short H uses its explicit zero-feature path')
    used=np.asarray(sorted(set(map(int,used))),dtype=np.int32)
    flags=np.asarray([np.any((used>=a)&(used<b)) for a,b in ((0,51),(51,77),(77,93),(93,117),(117,153))],dtype=np.bool_)
    engine=(*(e.args for e in feature_state.engines),flags,np.zeros(153,dtype=np.float32))
    normalization=(feature_state.center.copy(),feature_state.spread.copy(),used,np.zeros(153,dtype=np.float32))
    return engine,normalization
