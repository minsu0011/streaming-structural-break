"""Ten immutable outcome-blind H summaries and an unchanged J102 feature bank."""
from pathlib import Path
import hashlib
import json
import numpy as np
from numba import njit
from src.features.config import CONFIG
from src.features.streaming import feature_names as old_names,FEATURE_FAMILIES
from src.features.arch_input import ARCHCalibratedState
from src.next.fused_runtime import _arch_args,_arch_feature_step
from src.next.whitening import fit_historical
from src.next.fast_variance_initialization import fast_variance_parameters
from src.next2.dependence_runtime import historical_z
from src.next.serial_variance_evidence import SerialVarianceState,SerialVarianceConfig,serial_variance_step
from src.next.channel_evidence import CHANNEL_STATS

CONTEXT_NAMES=('log_H_std','log_H_tail_ratio','H_acf1','H_acf4','H_AR_coefficient1','H_AR_coefficient2',
    'H_ARCH_alpha','log_H_spectral_low_high','log_H_length','H_normalized_abs_tail_rate')
OLD_CONFIG=CONFIG.variant(normalization='median_mad',scales=(5,20,160))
OLD_COLUMNS=np.array([j for j,family in enumerate(FEATURE_FAMILIES) if family in 'ABCD'],dtype=np.int32)


def validate(settings):
    if settings!={'version':'historical_context_v1','kind':'base'}:raise ValueError('Exact H-context base settings required')


def context(historical):
    h=np.asarray(historical,dtype=np.float64)
    if h.ndim!=1 or len(h)<16 or not np.isfinite(h).all():raise ValueError('Finite clean historical vector of at least 16 points required')
    centered=h-np.median(h);abs_h=np.abs(centered);standard=max(float(h.std()),1e-12)
    q50,q95=np.quantile(abs_h,[.5,.95]);tail=max(float(q95),1e-12)/max(float(q50),1e-12)
    acfs=[]
    for lag in (1,4):
        a=h[lag:]-h[lag:].mean();b=h[:-lag]-h[:-lag].mean()
        acfs.append(float(np.clip((a@b)/max(float(np.sqrt((a@a)*(b@b))),1e-12),-1,1)))
    fit=fit_historical(h,8);residuals=fit['historical_innovations'];parameters,conditional=fast_variance_parameters(residuals,'arch1')
    variance=max(float(np.minimum((residuals-parameters[0])**2,parameters[5]).mean()),1e-12)
    z,replayed=historical_z(residuals,parameters,variance);np.testing.assert_array_equal(replayed,conditional)
    spectrum=np.abs(np.fft.rfft(h-h.mean()))**2;frequency=np.fft.rfftfreq(len(h))
    low=float(spectrum[(frequency>0)&(frequency<=.125)].sum());high=float(spectrum[frequency>.125].sum())
    values=[np.clip(np.log(standard),-12,12)/6,np.clip(np.log(max(tail,1e-12)),0,4)/2,*acfs,
        np.clip(fit['coefficients'][0],-2,2),np.clip(fit['coefficients'][1],-2,2),np.clip(parameters[3],0,1),
        np.clip(np.log((low+1e-12)/(high+1e-12)),-6,6)/3,np.log(len(h))/10,float(np.mean(np.abs(z)>2))]
    result=np.asarray(values,dtype=np.float32)
    if result.shape!=(10,) or not np.isfinite(result).all():raise RuntimeError('Invalid fixed H context')
    return result


@njit(cache=False)
def step(point,arch_args,serial_args,h_context,output):
    old=_arch_feature_step(point,arch_args)
    for j in range(51):output[j]=old[OLD_COLUMNS[j]]
    serial=serial_variance_step(point,*serial_args);output[51:77]=serial[:26]
    output[77:]=h_context
    return output


@njit(cache=False)
def replay(points,args):
    result=np.empty((len(points),87),dtype=np.float32)
    for j in range(len(points)):result[j]=step(float(points[j]),*args)
    return result


class HistoricalContextState:
    def __init__(self,historical,settings):
        validate(settings);assert len(OLD_COLUMNS)==51
        old=ARCHCalibratedState(historical,config=OLD_CONFIG)
        serial=SerialVarianceState(historical,SerialVarianceConfig(order=8,normalization='arch1',max_age=512))
        serial_args=(serial.filter_args,serial.parameters,serial.conditional,serial.mode,
            serial.calibration,serial.work,serial.evidence_args,serial.age_variances)
        self.args=(_arch_args(old),serial_args,context(historical),np.empty(87,dtype=np.float32))

    def update(self,point):return step(float(point),*self.args)

    def replay(self,points):return replay(np.asarray(points,dtype=np.float32),self.args)


def make_state(historical,settings):return HistoricalContextState(historical,settings)


def feature_names(settings):
    validate(settings);old=old_names(OLD_CONFIG)
    return tuple('arch8__'+old[j] for j in OLD_COLUMNS)+tuple('serial_variance__'+channel+'_'+stat
        for channel in ('raw_energy','log_variance_forecast') for stat in CHANNEL_STATS)+tuple('next4_context_'+name for name in CONTEXT_NAMES)


def implementation_hash(settings):
    validate(settings);root=Path(__file__).resolve().parents[2]
    paths=[Path(__file__),root/'src/next/fused_runtime.py',root/'src/next/serial_variance_evidence.py',
        root/'src/next/variance_evidence.py',root/'src/next/normalization.py',root/'src/next/whitening.py',
        root/'src/next/fast_variance_initialization.py',root/'src/next2/dependence_runtime.py',
        root/'src/next2/complement_bank.py',root/'src/next/serial_channel_evidence.py',root/'src/next/channel_evidence.py']
    paths+=sorted((root/'src/features').glob('*.py'))
    paths+=[root/'configs'/name for name in ('features.json','arch_input.json','ar_residual_input.json','historical_context.json','historical_quantiles.json')]
    return hashlib.sha256(b''.join(p.read_bytes() for p in paths)+json.dumps(settings,sort_keys=True).encode()).hexdigest()
