"""Append separate robust lower/upper energy evidence to the unchanged J102 bank."""
from pathlib import Path
import hashlib
import json
import numpy as np
from numba import njit
from src.features.config import CONFIG
from src.features.streaming import feature_names as old_names,FEATURE_FAMILIES
from src.features.arch_input import ARCHCalibratedState
from src.next.fused_runtime import _arch_args,_arch_feature_step
from src.next.whitening import fit_historical,innovation_update
from src.next.normalization import fit_normalization
from src.next.variance_evidence import historical_variance_channels,variance_channel_update
from src.next.serial_variance_evidence import SerialVarianceState,SerialVarianceConfig,serial_variance_step
from src.next.channel_evidence import CHANNEL_STATS

KINDS=('split_energy','log_semitail')
OLD_CONFIG=CONFIG.variant(normalization='median_mad',scales=(5,20,160))
OLD_COLUMNS=np.asarray([i for i,family in enumerate(FEATURE_FAMILIES) if family in 'ABCD'],dtype=np.int32)


def validate(settings):
    if set(settings)!={'version','kind'} or settings['version']!='separate_energy_branches_v1' or settings['kind'] not in KINDS:
        raise ValueError('Exact scale asymmetry contract required')


def branch_reference(channels,kind):
    energies=np.maximum(channels[:,[0,2]]+1.,0.)
    values=np.log(energies+1e-4) if kind=='log_semitail' else energies
    parameters=np.empty((2,7));branches=np.empty((len(values),4))
    for j in range(2):
        x=values[:,j];center=float(np.median(x));lower=np.maximum(center-x,0.);upper=np.maximum(x-center,0.)
        low_scale=max(float(np.median(lower[lower>0])) if np.any(lower>0) else 0.,1e-4)
        high_scale=max(float(np.median(upper[upper>0])) if np.any(upper>0) else 0.,1e-4)
        low=np.clip(lower/low_scale,0,12);high=np.clip(upper/high_scale,0,12)
        means=(float(low.mean()),float(high.mean()))
        spreads=(max(float(np.median(np.abs(low-means[0]))),.1),max(float(np.median(np.abs(high-means[1]))),.1))
        parameters[j]=[center,low_scale,high_scale,*means,*spreads]
        branches[:,j*2]=low;branches[:,j*2+1]=high
    return parameters,branches


@njit(cache=False)
def branch_values(raw,reference,kind,output):
    for j,index in enumerate((0,2)):
        value=max(raw[index]+1.,0.)
        if kind==1:value=np.log(value+1e-4)
        delta=value-reference[j,0]
        output[j*2]=min(max(-delta/reference[j,1],0.),12.)
        output[j*2+1]=min(max(delta/reference[j,2],0.),12.)
    return output


@njit(cache=False)
def appended_step(point,old_args,serial_args,branch_args,output):
    old=_arch_feature_step(point,old_args)
    for j in range(len(OLD_COLUMNS)):output[j]=old[OLD_COLUMNS[j]]
    serial=serial_variance_step(point,*serial_args)
    output[51:77]=serial[:26]
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
    return output


@njit(cache=False)
def replay(points,args):
    result=np.empty((len(points),93),dtype=np.float32)
    for j in range(len(points)):result[j]=appended_step(float(points[j]),*args)
    return result


class ScaleAsymmetryState:
    def __init__(self,historical,settings):
        validate(settings)
        if len(OLD_COLUMNS)!=51:raise RuntimeError('Original 51 ABCD feature columns required')
        old=ARCHCalibratedState(historical,config=OLD_CONFIG)
        original=SerialVarianceState(historical,SerialVarianceConfig(order=8,normalization='arch1',max_age=512))
        serial_args=(original.filter_args,original.parameters,original.conditional,original.mode,
            original.calibration,original.work,original.evidence_args,original.age_variances)
        fit=fit_historical(historical,8);profile=fit_normalization(fit['historical_innovations'],'arch1')
        parameters=profile['parameters'];variance=parameters[2]/max(1.-parameters[3]-parameters[4],.05)
        conditional=np.full(3,variance)
        channels=historical_variance_channels(fit['historical_innovations'],parameters,conditional,profile['mode'])
        np.testing.assert_allclose(conditional,profile['conditional'],rtol=1e-14,atol=1e-14)
        channels=channels[min(160,len(channels)//4):]
        reference,_=branch_reference(channels,settings['kind'])
        branch_args=(tuple(fit[k] for k in ('reference','coefficients','ring','counter')),parameters,
            profile['conditional'].copy(),profile['mode'],reference,KINDS.index(settings['kind']),
            np.empty(6),np.empty(4),np.zeros((4,2)),np.zeros(4),np.zeros(1,dtype=np.int64))
        self.args=(_arch_args(old),serial_args,branch_args,np.empty(93,dtype=np.float32))

    def update(self,point):return appended_step(float(point),*self.args)

    def replay(self,points):return replay(np.asarray(points,dtype=np.float32),self.args)


def make_state(historical,settings):return ScaleAsymmetryState(historical,settings)


def feature_names(settings):
    validate(settings)
    names=old_names(OLD_CONFIG)
    original=tuple('arch8__'+names[j] for j in OLD_COLUMNS)+tuple('serial_variance__'+channel+'_'+stat
        for channel in ('raw_energy','log_variance_forecast') for stat in CHANNEL_STATS)
    extra=tuple('next4_'+settings['kind']+'_'+channel+'_'+direction+'_'+stat
        for channel in ('raw_energy','conditional_energy') for direction in ('lower','upper')
        for stat in ('current','ewma8','ewma32','cusum'))
    return original+extra


def implementation_hash(settings):
    validate(settings);root=Path(__file__).resolve().parents[2]
    source=[Path(__file__),root/'src/next/fused_runtime.py',root/'src/next/serial_variance_evidence.py',
        root/'src/next/variance_evidence.py',root/'src/next/normalization.py',root/'src/next/whitening.py',
        root/'src/next/serial_channel_evidence.py',root/'src/next/channel_evidence.py']
    # Include all old feature primitives and their frozen JSON configurations.
    source+=sorted((root/'src/features').glob('*.py'))
    source+=[root/'configs'/name for name in ('features.json','arch_input.json','ar_residual_input.json',
        'historical_context.json','historical_quantiles.json')]
    return hashlib.sha256(b''.join(p.read_bytes() for p in source)+json.dumps(settings,sort_keys=True).encode()).hexdigest()
