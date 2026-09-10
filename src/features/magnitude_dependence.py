"""Six causal magnitude-dependence features beside unchanged AR8 ABCD evidence."""
import hashlib,json
from pathlib import Path
import numpy as np
from numba import njit
from src.features.config import CONFIG
from src.features.streaming import StreamingFeatureState,feature_names
from src.features.ar_order_input import historical_filter,implementation_hash as ar_hash
from src.features.ar_residual_input import innovation_one
from src.features.historical_calibration import CalibrationPolicy,HistoricalCalibratedState,_transform

MAGNITUDE_POLICY=json.loads((Path(__file__).resolve().parents[2]/'configs/magnitude_dependence.json').read_text(encoding='utf-8'))


def names(config=CONFIG):
    return feature_names(config)[:51]+tuple(f'{kind}_lag1_evidence_{scale}' for scale in config.scales for kind in ('absolute','squared'))


def model_families(families):
    if families=='ABCDV':return 'ABCDEFQ'
    if families=='V':return 'EFQ'
    raise ValueError('Unfrozen magnitude-dependence family scope')


def implementation_hash(policy,config=CONFIG):
    return hashlib.sha256(Path(__file__).read_bytes()+json.dumps(MAGNITUDE_POLICY,sort_keys=True).encode()+ar_hash(policy,config,8).encode()).hexdigest()


@njit(cache=False)
def update_magnitude(residual,reference,baseline,previous,ew,counters,alphas,output):
    residual=np.float64(residual)
    location,scale,clip,absolute_mean,absolute_sd,squared_mean,squared_sd,cap=reference
    z=min(max(((residual if np.isfinite(residual) else location)-location)/scale,-clip),clip)
    absolute=(abs(z)-absolute_mean)/absolute_sd;squared=(min(z*z,cap)-squared_mean)/squared_sd
    for k in range(3):
        counters[k]=(1.-alphas[k])*counters[k]+1.;root=np.sqrt(counters[k])
        ew[k,0]+=alphas[k]*(absolute*previous[0]-ew[k,0]);ew[k,1]+=alphas[k]*(squared*previous[1]-ew[k,1])
        output[k*2]=(ew[k,0]-baseline[0])*root;output[k*2+1]=(ew[k,1]-baseline[1])*root
    previous[0]=absolute;previous[1]=squared


@njit(cache=False)
def replay_magnitude(points,args):
    result=np.empty((len(points),6),dtype=np.float32)
    for k in range(len(points)):
        update_magnitude(np.float64(points[k]),*args);result[k]=args[-1]
    return result


class MagnitudeFeatureState:
    def __init__(self,historical_residual,*,config=CONFIG,policy=CalibrationPolicy(method='historical_median_mad')):
        h=np.asarray(historical_residual,dtype=np.float32);base=StreamingFeatureState(h,config=config);location,scale=base.reference[:2]
        z=np.clip((np.where(np.isfinite(h),h,location).astype(np.float64)-location)/scale,-config.clip_z,config.clip_z)
        if not len(z):z=np.zeros(1,dtype=np.float64)
        absolute=abs(z);cap=max(float(np.quantile(z*z,MAGNITUDE_POLICY['squared_residual_winsor_quantile'])),config.moment_floor);squared=np.minimum(z*z,cap)
        mean_a=float(absolute.mean());sd_a=np.sqrt(max(float(absolute.var()),config.moment_floor))
        mean_q=float(squared.mean());sd_q=np.sqrt(max(float(squared.var()),config.moment_floor))
        a=(absolute-mean_a)/sd_a;q=(squared-mean_q)/sd_q
        reference=np.array([location,scale,config.clip_z,mean_a,sd_a,mean_q,sd_q,cap])
        baseline=np.array([np.mean(a[1:]*a[:-1]),np.mean(q[1:]*q[:-1])]) if len(a)>1 else np.zeros(2)
        self.args=(reference,baseline,np.array([a[-1],q[-1]]),np.tile(baseline,(3,1)),np.zeros(3),np.asarray(config.alphas),np.zeros(6,dtype=np.float32))
        reference_rows=replay_magnitude(h,tuple(value.copy() for value in self.args))
        if not len(reference_rows):reference_rows=np.zeros((1,6),dtype=np.float32)
        burn=min(policy.burn_in,max(0,len(reference_rows)//2));reference_rows=reference_rows[burn:].astype(np.float64)
        self.center=np.median(reference_rows,axis=0);self.scale=np.maximum(np.median(abs(reference_rows-self.center),axis=0)*1.4826,policy.standard_deviation_floor)
        self.mask=np.ones(6,dtype=np.bool_);self.output=np.zeros(6,dtype=np.float32);self.policy=policy
    def update_and_get(self,residual):
        update_magnitude(float(residual),*self.args);_transform(self.args[-1],self.center,self.scale,self.mask,float(self.policy.clip),self.output)
        return self.output
    @property
    def state_array_bytes(self):return sum(x.nbytes for x in (*self.args,self.center,self.scale,self.mask,self.output))


class MagnitudeARState:
    def __init__(self,historical,*,config=CONFIG,policy=CalibrationPolicy(method='historical_median_mad'),include_abcd=True):
        self.reference,self.coefficients,self.ring,residual=historical_filter(historical,config,8);self.counter=np.zeros(1,dtype=np.int64)
        self.magnitude=MagnitudeFeatureState(residual,config=config,policy=policy)
        self.inner=HistoricalCalibratedState(residual,config=config,policy=policy) if include_abcd else None
        self.output=np.zeros(57 if include_abcd else 6,dtype=np.float32)
    def update_and_get(self,point):
        residual=innovation_one(float(point),self.reference,self.coefficients,self.ring,self.counter)
        if self.inner is not None:self.output[:51]=self.inner.update_and_get(residual)[:51]
        self.output[-6:]=self.magnitude.update_and_get(residual)
        return self.output
    @property
    def state_array_bytes(self):
        return self.output.nbytes+self.magnitude.state_array_bytes+(self.inner.state_array_bytes if self.inner is not None else 0)+sum(x.nbytes for x in (self.reference,self.coefficients,self.ring,self.counter))
