"""Historical feature-quantile mapping; the AR8 input and 57 raw features stay fixed."""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.special import ndtri
from numba import njit
from src.features.config import CONFIG
from src.features.streaming import StreamingFeatureState,FEATURE_FAMILIES
from src.features.historical_calibration import CalibrationPolicy,_historical_replay
from src.features.ar_order_input import historical_filter,implementation_hash as ar_hash
from src.features.ar_residual_input import innovation_one

QUANTILE_POLICY=json.loads((Path(__file__).resolve().parents[2]/'configs/historical_quantiles.json').read_text(encoding='utf-8'))


def implementation_hash(families,policy,config=CONFIG):
    return hashlib.sha256(Path(__file__).read_bytes()+families.encode()+json.dumps(QUANTILE_POLICY,sort_keys=True).encode()+ar_hash(policy,config,8).encode()).hexdigest()


def historical_mapping(historical,*,families,config=CONFIG,policy=CalibrationPolicy(method='historical_median_mad')):
    if families not in QUANTILE_POLICY['family_scopes']:raise ValueError('Unfrozen historical quantile scope')
    if policy.method!='historical_median_mad':raise ValueError('The quantile comparison retains the original MAD calibration elsewhere')
    h=np.asarray(historical,dtype=np.float32);state=StreamingFeatureState(h,config=config)
    reference=_historical_replay(h,state.reference,state.baseline,state.thresholds,state.ring,state.ew,state.counters,state.output,state.parameters,state.alphas,state.lags)
    if not len(reference):reference=np.zeros((1,len(FEATURE_FAMILIES)),dtype=np.float32)
    burn=min(policy.burn_in,max(0,len(reference)//2));reference=reference[burn:].astype(np.float64)
    center=np.median(reference,axis=0);scale=np.maximum(np.median(abs(reference-center),axis=0)*1.4826,policy.standard_deviation_floor)
    mask=np.array([family in policy.signed_families for family in FEATURE_FAMILIES]);mask[QUANTILE_POLICY['nonstationary_columns_excluded']]=False
    probabilities=np.array(QUANTILE_POLICY['probabilities']);knots=np.zeros((len(center),len(probabilities)));scores=knots.copy();lengths=np.zeros(len(center),dtype=np.int32)
    for j,family in enumerate(FEATURE_FAMILIES):
        if not mask[j] or family not in families:continue
        values=np.sort(reference[:,j]);unique=np.unique(np.quantile(values,probabilities))
        if len(unique)<2:continue
        left=np.searchsorted(values,unique,side='left');right=np.searchsorted(values,unique,side='right')
        mid_cdf=(left+right)/(2.*len(values));mid_cdf=np.clip(mid_cdf,probabilities[0],probabilities[-1])
        count=len(unique);knots[j,:count]=unique;scores[j,:count]=ndtri(mid_cdf);lengths[j]=count
    return center,scale,mask,knots,scores,lengths


@njit(cache=False)
def quantile_transform(source,center,scale,mask,knots,scores,lengths,clip,output):
    for j in range(len(source)):
        count=lengths[j]
        if count>=2:
            value=np.float64(source[j]);k=np.searchsorted(knots[j,:count],value,side='left')
            if k==0:output[j]=scores[j,0]
            elif k==count:output[j]=scores[j,count-1]
            else:
                fraction=(value-knots[j,k-1])/(knots[j,k]-knots[j,k-1])
                output[j]=scores[j,k-1]+fraction*(scores[j,k]-scores[j,k-1])
        elif mask[j]:output[j]=min(max((np.float64(source[j])-center[j])/scale[j],-clip),clip)
        else:output[j]=source[j]


class HistoricalQuantileARState:
    def __init__(self,historical,*,families='C',config=CONFIG,policy=CalibrationPolicy(method='historical_median_mad')):
        self.reference,self.coefficients,self.ring,residual=historical_filter(historical,config,8);self.counter=np.zeros(1,dtype=np.int64)
        self.state=StreamingFeatureState(residual,config=config);self.mapping=historical_mapping(residual,families=families,config=config,policy=policy)
        self.policy=policy;self.output=np.zeros(len(FEATURE_FAMILIES),dtype=np.float32)
    def update_and_get(self,point):
        residual=innovation_one(float(point),self.reference,self.coefficients,self.ring,self.counter)
        source=self.state.update_and_get(residual);quantile_transform(source,*self.mapping,float(self.policy.clip),self.output)
        return self.output
    @property
    def state_array_bytes(self):
        return self.state.state_array_bytes+self.output.nbytes+sum(x.nbytes for x in (self.reference,self.coefficients,self.ring,self.counter,*self.mapping))
