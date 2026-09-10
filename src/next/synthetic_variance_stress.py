"""Controlled model diagnostics, separate from competition DEV selection."""
from pathlib import Path
import copy
import hashlib
import numpy as np
from numba import njit,typeof
from src.next.fused_runtime import _resident_class
from src.streaming.fused_cross_order import fused_equal_average
from src.streaming.fused_arch import fused_arch_predict


REGIMES = (
    {'name':'IID_GAUSSIAN','phi':0.,'arch_alpha':0.,'tail_df':0},
    {'name':'AR06_GAUSSIAN','phi':.6,'arch_alpha':0.,'tail_df':0},
    {'name':'AR09_STUDENT5','phi':.9,'arch_alpha':0.,'tail_df':5},
    {'name':'AR03_ARCH065_GAUSSIAN','phi':.3,'arch_alpha':.65,'tail_df':0},
    {'name':'AR_MINUS04_ARCH04_STUDENT5','phi':-.4,'arch_alpha':.4,'tail_df':5})
SCENARIOS = ('NO_CHANGE','MEAN_SHIFT','SCALE_SHIFT','DEPENDENCE_SHIFT','TAIL_SHIFT','VARIANCE_PERSISTENCE_SHIFT','MIXED_SHIFT')


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@njit(cache=True)
def simulate_segment(noise,y,previous_innovation,phi,alpha,original_phi,tau,scenario,polarity):
    output = np.empty(len(noise),dtype=np.float32)
    sigma = 1./np.sqrt(1.-original_phi*original_phi)
    for t in range(len(noise)):
        changed = tau>=0 and t>=tau
        current_phi,current_alpha = phi,alpha
        mean,scale = 0.,1.
        if changed:
            if scenario in (1,6):
                mean = polarity*.5*sigma
            if scenario in (2,6):
                scale = 1.5 if polarity>0 else 1./1.5
            if scenario in (3,6):
                current_phi = -.6 if original_phi>=0 else .6
            if scenario==5:
                current_alpha = .85
        variance = (1.-current_alpha)+current_alpha*previous_innovation*previous_innovation
        innovation = np.sqrt(max(variance,1e-12))*noise[t]
        # Dependence changes preserve stationary marginal second moment in the
        # Gaussian/non-ARCH case, reducing scale/dependence confounding.
        correction = np.sqrt((1.-current_phi*current_phi)/(1.-original_phi*original_phi))
        y = current_phi*y+correction*innovation
        previous_innovation = innovation
        output[t] = scale*y+mean
    return output,y,previous_innovation


def generate_series(policy,partition,regime_index,index):
    regime = policy['regimes'][regime_index]
    rng = np.random.default_rng(np.random.SeedSequence([policy['seed'],partition,regime_index,index]))
    total = policy['burn_points']+policy['historical_points']+policy['online_points']
    gaussian = rng.normal(size=total)
    student5 = rng.standard_t(5,size=total)*np.sqrt(3./5.)
    student3 = rng.standard_t(3,size=total)/np.sqrt(3.)
    base = gaussian if regime['tail_df']==0 else student5
    cut = policy['burn_points']+policy['historical_points']
    prefix,y,e = simulate_segment(base[:cut],0.,0.,regime['phi'],regime['arch_alpha'],regime['phi'],-1,0,1)
    historical = prefix[policy['burn_points']:]
    tau = int(policy['break_times'][index%len(policy['break_times'])])
    polarity = 1 if index%2==0 else -1
    scenarios = policy['scenarios'] if partition==1 else ['NO_CHANGE']
    online = {}
    for name in scenarios:
        code = SCENARIOS.index(name)
        noise = base[cut:].copy()
        if name=='TAIL_SHIFT':
            noise[tau:] = student3[cut+tau:]
        points,_,_ = simulate_segment(noise,y,e,regime['phi'],regime['arch_alpha'],regime['phi'],tau if code else -1,code,polarity)
        online[name] = points
    if not np.isfinite(historical).all() or any(not np.isfinite(values).all() for values in online.values()):
        raise RuntimeError('Synthetic process became nonfinite')
    return historical,online,tau,polarity


@njit(cache=False)
def cross_kernel(point,args):
    return fused_equal_average(point,args[0],args[1])


@njit(cache=False)
def arch_kernel(point,args):
    return np.float32(fused_arch_predict(point,*args))


class HistoricalTemplate:
    """Replay distinct synthetic futures from identical fresh H-only state."""
    def __init__(self,prepared,historical,role):
        self.detector = prepared.new_state(historical)
        if role=='PRIMARY':
            self.args,self.kernel = self.detector.call_args,cross_kernel
        elif role=='HIGHEST_MEAN':
            d = self.detector
            self.args = d.ar_args,d.variance_args,d.state_args,d.calibration_args,d.prepared.tree_args,d.prepared.is_xgboost
            self.kernel = arch_kernel
        else:
            self.args,self.kernel = self.detector.call_args,self.detector.kernel

    def fresh_resident(self):
        # Copy every nested array, including immutable trees, to make absence
        # of shared mutable state explicit. The H fit itself is not repeated.
        args = copy.deepcopy(self.args)
        return _resident_class(self.kernel,typeof(args))(args)

    def predict(self,points):
        resident = self.fresh_resident()
        return np.asarray([resident.predict_one(float(point)) for point in points],dtype=np.float32)
