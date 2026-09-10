"""Three complementary, fixed-memory detectors; parameters frozen before CV."""
from dataclasses import dataclass,asdict
import hashlib,json
from pathlib import Path
import numpy as np
from numba import njit

@dataclass(frozen=True)
class SequentialPolicy:
    mean_alternative: float = .5
    scale_alternatives: tuple = (.5,2.0)
    tail_quantiles: tuple = (.05,.95)
    tail_probability_multipliers: tuple = (.5,2.0)
    evidence_decay: float = .995
    residual_clip: float = 6.0
    phi_limit: float = .99
    floor: float = 1e-8

POLICY=SequentialPolicy()
NAMES=('C1_ar_residual_mean_lr','C2_ar_residual_scale_lr','C3_quantile_tail_lr')

def implementation_hash(policy=POLICY):
    return hashlib.sha256(Path(__file__).read_bytes()+json.dumps(asdict(policy),sort_keys=True).encode()).hexdigest()

@njit(cache=False)
def update(point,reference,parameters,evidence,output):
    x=point if np.isfinite(point) else reference[0]
    residual=(x-reference[0])-reference[2]*(reference[5]-reference[0])
    z=(residual-reference[3])/reference[4]
    z=min(max(z,-parameters[0]),parameters[0])
    delta=parameters[1]
    increments=np.empty(6,dtype=np.float64)
    increments[0]=delta*z-.5*delta*delta
    increments[1]=-delta*z-.5*delta*delta
    for j in range(2):
        ratio=parameters[3+j]
        # Historical residual energy is used as the null reference; clipping is
        # calibrated by its historical mean so rare spikes do not add null drift.
        increments[2+j]=-.5*np.log(ratio)+.5*(1-1/ratio)*(z*z/reference[8])
    exceed=(x<reference[6]) or (x>reference[7])
    p0=reference[9]
    for j in range(2):
        p1=parameters[5+j]
        increments[4+j]=np.log(p1/p0) if exceed else np.log((1-p1)/(1-p0))
    for j in range(6):evidence[j]=max(0.,parameters[2]*evidence[j]+increments[j])
    for j in range(3):
        score=max(evidence[2*j],evidence[2*j+1])
        output[j]=score/(10.+score)
    reference[5]=x

class SequentialDetectors:
    def __init__(self,historical,*,policy=POLICY):
        h=np.asarray(historical,dtype=np.float64)
        h=h[np.isfinite(h)]
        if not len(h):h=np.zeros(2)
        mean=float(h.mean());center=h-mean
        phi=float(np.dot(center[1:],center[:-1])/max(np.dot(center[:-1],center[:-1]),policy.floor)) if len(h)>1 else 0.
        phi=float(np.clip(phi,-policy.phi_limit,policy.phi_limit))
        residual=center[1:]-phi*center[:-1] if len(h)>1 else center
        rmean=float(residual.mean());rscale=max(float(residual.std()),policy.floor)
        z=np.clip((residual-rmean)/rscale,-policy.residual_clip,policy.residual_clip)
        energy=max(float(np.mean(z*z)),policy.floor)
        low,high=np.quantile(h,policy.tail_quantiles)
        p0=float(np.clip(np.mean((h<low)|(h>high)),.001,.999))
        alternatives=[float(np.clip(p0*m,.001,.999)) for m in policy.tail_probability_multipliers]
        self.reference=np.array([mean,max(float(h.std()),policy.floor),phi,rmean,rscale,h[-1],low,high,energy,p0],dtype=np.float64)
        self.parameters=np.array([policy.residual_clip,policy.mean_alternative,policy.evidence_decay,*policy.scale_alternatives,*alternatives],dtype=np.float64)
        self.evidence=np.zeros(6,dtype=np.float64);self.output=np.zeros(3,dtype=np.float32)

    def update_and_get(self,point):
        update(float(point),self.reference,self.parameters,self.evidence,self.output)
        return self.output

    @property
    def state_array_bytes(self):return sum(value.nbytes for value in vars(self).values() if isinstance(value,np.ndarray))
