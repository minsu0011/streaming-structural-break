"""Append H-only asymmetric or lower-quartile summaries to exact J102 inputs."""
from pathlib import Path
import hashlib
import json
import numpy as np
from numba import njit
from src.next.whitening import fit_historical,innovation_update
from src.next.normalization import fit_normalization
from src.next.variance_evidence import historical_variance_channels,variance_channel_update
from src.next.serial_variance_evidence import SerialVarianceState,SerialVarianceConfig,serial_variance_step
from src.next.serial_channel_evidence import historical_correlations,sum_variances,serial_channel_step
from src.next.channel_evidence import CHANNEL_STATS,make_channel_args

KINDS=('S1_LOWER_SEMISCALE','S2_LOWER_QUANTILE')
EXTRA_STATS=('current','ewma8','ewma32','bayes1')
EXTRA_COLUMNS=np.asarray([c*13+CHANNEL_STATS.index(s) for c in range(2) for s in EXTRA_STATS],dtype=np.int32)


def validate(settings):
    if set(settings)!={'kind','version'} or settings['kind'] not in KINDS or settings['version']!='scale_calibration_append_v1':
        raise ValueError('Frozen scale calibration append contract required')


def fit_transform(channels,kind):
    channels=np.asarray(channels,dtype=np.float64)
    if channels.ndim!=2 or channels.shape[1]!=2 or not np.isfinite(channels).all():raise ValueError('Two finite H channels required')
    if kind==KINDS[0]:
        center=channels.mean(axis=0);delta=channels-center
        lower=np.sqrt(np.mean(np.minimum(delta,0.)**2,axis=0)/np.maximum(np.mean(delta<0,axis=0),1./len(delta)))
        upper=np.sqrt(np.mean(np.maximum(delta,0.)**2,axis=0)/np.maximum(np.mean(delta>=0,axis=0),1./len(delta)))
        lower=np.maximum(lower,1e-6);upper=np.maximum(upper,1e-6)
        transformed=delta/np.where(delta<0,lower,upper)
        parameters=np.stack([center,lower,upper,transformed.mean(axis=0),np.maximum(transformed.std(axis=0),1e-6)])
        standardized=(transformed-parameters[3])/parameters[4]
        return parameters,standardized
    if kind==KINDS[1]:
        threshold=np.quantile(channels,.25,axis=0,method='linear')
        hit=(channels<=threshold).astype(np.float64)
        center=hit.mean(axis=0);scale=np.maximum(hit.std(axis=0),1e-6)
        parameters=np.stack([threshold,center,scale,np.zeros(2),np.zeros(2)])
        return parameters,(hit-center)/scale
    raise ValueError('Unknown fixed calibration')


@njit(cache=False)
def transform_point(raw,parameters,kind,output):
    for j in range(2):
        if kind==0:
            delta=raw[j]-parameters[0,j]
            value=delta/(parameters[1,j] if delta<0. else parameters[2,j])
            output[j]=(value-parameters[3,j])/parameters[4,j]
        else:
            hit=1. if raw[j]<=parameters[0,j] else 0.
            output[j]=(hit-parameters[1,j])/parameters[2,j]
    return output


@njit(cache=False)
def appended_step(point,original_args,new_args,output):
    original=serial_variance_step(point,*original_args)
    output[:26]=original[:26]
    filter_args,parameters,conditional,mode,transform,kind,raw,work,evidence,variances=new_args
    residual=innovation_update(point,*filter_args)
    variance_channel_update(residual,parameters,conditional,mode,raw)
    transform_point(raw,transform,kind,work)
    values=serial_channel_step(work,evidence,variances)
    for j in range(8):output[26+j]=values[EXTRA_COLUMNS[j]]
    return output


@njit(cache=False)
def replay(points,args):
    output=np.empty((len(points),34),dtype=np.float32)
    for j in range(len(points)):output[j]=appended_step(float(points[j]),*args)
    return output


class ScaleCalibrationState:
    def __init__(self,historical,settings):
        validate(settings)
        original=SerialVarianceState(historical,SerialVarianceConfig(order=8,normalization='arch1',max_age=512))
        original_args=(original.filter_args,original.parameters,original.conditional,original.mode,
            original.calibration,original.work,original.evidence_args,original.age_variances)
        fit=fit_historical(historical,8);profile=fit_normalization(fit['historical_innovations'],'arch1')
        parameters=profile['parameters'];variance=parameters[2]/max(1.-parameters[3]-parameters[4],.05)
        conditional=np.full(3,variance)
        raw=historical_variance_channels(fit['historical_innovations'],parameters,conditional,profile['mode'])
        np.testing.assert_allclose(conditional,profile['conditional'],rtol=1e-14,atol=1e-14)
        raw=raw[min(160,len(raw)//4):,:2]
        transform,standardized=fit_transform(raw,settings['kind'])
        evidence=make_channel_args(2,512)
        variances=sum_variances(evidence[0],historical_correlations(standardized))
        new_args=(tuple(fit[k] for k in ('reference','coefficients','ring','counter')),parameters,profile['conditional'].copy(),
            profile['mode'],transform,KINDS.index(settings['kind']),np.empty(6),np.empty(2),evidence,variances)
        self.args=(original_args,new_args,np.empty(34,dtype=np.float32))

    def update(self,point):return appended_step(float(point),*self.args)

    def replay(self,points):return replay(np.asarray(points,dtype=np.float32),self.args)


def make_state(historical,settings):return ScaleCalibrationState(historical,settings)


def feature_names(settings):
    validate(settings)
    original=tuple(channel+'_'+stat for channel in ('raw_energy','log_variance_forecast') for stat in CHANNEL_STATS)
    extra=tuple('calibrated_'+channel+'_'+stat for channel in ('raw_energy','log_variance_forecast') for stat in EXTRA_STATS)
    return original+extra


def feature_groups(settings):
    validate(settings);return ('NEXT3_SCALE_CALIBRATION',)*34


def implementation_hash(settings):
    validate(settings);root=Path(__file__).resolve().parents[2]
    files=('src/next3/scale_calibration.py','src/next/next3_scale_calibration.py','src/next/serial_variance_evidence.py',
        'src/next/variance_evidence.py','src/next/normalization.py','src/next/serial_channel_evidence.py',
        'src/next/channel_evidence.py','src/next/whitening.py')
    return hashlib.sha256(b''.join((root/p).read_bytes() for p in files)+json.dumps(settings,sort_keys=True).encode()).hexdigest()
