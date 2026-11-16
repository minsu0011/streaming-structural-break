"""Immutable H masses on fixed residual cells; constant-memory divergences."""
from pathlib import Path
import hashlib
import json
import numpy as np
from numba import njit
from src.next.whitening import fit_historical,innovation_update
from src.next.fast_variance_initialization import fast_variance_parameters
from src.next2.dependence_runtime import historical_z
from src.next2.complement_bank import normalized_one

KINDS=('hellinger','js','shape_ratio')
EDGES=np.array([-2.,-1.,-.5,0.,.5,1.,2.])
HALVES=(8.,32.,128.)
DISTANCE_NAMES=('total_divergence','negative_divergence','positive_divergence','center_divergence',
    'moderate_divergence','extreme_divergence','signed_extreme_delta','center_mass_delta')
SHAPE_NAMES=('center_to_moderate_log_ratio','extreme_to_moderate_log_ratio','extreme_asymmetry_delta',
    'moderate_asymmetry_delta','center_mass_delta','extreme_mass_delta','cell_mean_delta','total_variation')


def validate(settings):
    if set(settings)!={'version','kind'} or settings['version']!='fixed_residual_cells_v1' or settings['kind'] not in KINDS:
        raise ValueError('Exact frozen fixed-bin settings required')


@njit(cache=False)
def summaries(mass,reference,mode):
    out=np.empty(8)
    center=mass[3]+mass[4];center0=reference[3]+reference[4]
    moderate=mass[1]+mass[2]+mass[5]+mass[6];moderate0=reference[1]+reference[2]+reference[5]+reference[6]
    extreme=mass[0]+mass[7];extreme0=reference[0]+reference[7]
    if mode<2:
        terms=np.empty(8)
        for j in range(8):
            p=max(mass[j],1e-12);q=reference[j]
            if mode==0:terms[j]=.5*(np.sqrt(p)-np.sqrt(q))**2
            else:
                mid=.5*(p+q)
                terms[j]=.5*(p*np.log(p/mid)+q*np.log(q/mid))
        out[0]=terms.sum();out[1]=terms[:4].sum();out[2]=terms[4:].sum()
        out[3]=terms[3]+terms[4];out[4]=terms[1]+terms[2]+terms[5]+terms[6];out[5]=terms[0]+terms[7]
        out[6]=(mass[7]-mass[0])-(reference[7]-reference[0]);out[7]=center-center0
    else:
        eps=1e-6
        out[0]=np.log((center+eps)/(moderate+eps))-np.log((center0+eps)/(moderate0+eps))
        out[1]=np.log((extreme+eps)/(moderate+eps))-np.log((extreme0+eps)/(moderate0+eps))
        out[2]=(mass[7]-mass[0])/(extreme+eps)-(reference[7]-reference[0])/(extreme0+eps)
        out[3]=((mass[5]+mass[6])-(mass[1]+mass[2]))/(moderate+eps)-((reference[5]+reference[6])-(reference[1]+reference[2]))/(moderate0+eps)
        out[4]=center-center0;out[5]=extreme-extreme0
        out[6]=0.;out[7]=0.
        for j in range(8):
            out[6]+=(j-3.5)*(mass[j]-reference[j]);out[7]+=.5*abs(mass[j]-reference[j])
    return out


@njit(cache=False)
def step(point,filter_args,parameters,conditional,reference,masses,alphas,mode,output):
    z=normalized_one(innovation_update(point,*filter_args),parameters,conditional)
    cell=np.searchsorted(EDGES,z,side='right')
    for half in range(3):
        masses[half]*=1.-alphas[half]
        masses[half,cell]+=alphas[half]
        output[half*8:half*8+8]=summaries(masses[half],reference,mode)
    output[24]=-np.log(reference[cell]);output[25]=(cell-3.5)/3.5
    return output


@njit(cache=False)
def replay(points,args):
    result=np.empty((len(points),26),dtype=np.float32)
    for j in range(len(points)):result[j]=step(float(points[j]),*args)
    return result


class DistributionState:
    def __init__(self,historical,settings):
        validate(settings);fit=fit_historical(historical,8);residuals=fit['historical_innovations']
        parameters,conditional=fast_variance_parameters(residuals,'arch1')
        variance=max(float(np.minimum((residuals-parameters[0])**2,parameters[5]).mean()),1e-12)
        z,replayed=historical_z(residuals,parameters,variance)
        np.testing.assert_array_equal(replayed,conditional)
        counts=np.bincount(np.searchsorted(EDGES,z,side='right'),minlength=8).astype(float)+.5
        reference=counts/counts.sum()
        self.args=(tuple(fit[k] for k in ('reference','coefficients','ring','counter')),parameters,conditional,
            reference,np.tile(reference,(3,1)),np.array([1.-np.exp(-np.log(2.)/h) for h in HALVES]),
            KINDS.index(settings['kind']),np.empty(26,dtype=np.float32))

    def update(self,point):return step(float(point),*self.args)

    def replay(self,points):return replay(np.asarray(points,dtype=np.float32),self.args)


def make_state(historical,settings):return DistributionState(historical,settings)


def feature_names(settings):
    validate(settings);names=SHAPE_NAMES if settings['kind']=='shape_ratio' else DISTANCE_NAMES
    return tuple('next4_'+settings['kind']+'_h'+str(int(half))+'_'+name for half in HALVES for name in names)+(
        'next4_H_cell_negative_log_mass','next4_current_cell_signed_position')


def implementation_hash(settings):
    validate(settings);root=Path(__file__).resolve().parents[2]
    paths=['src/next4/distribution.py','src/next/whitening.py','src/next/fast_variance_initialization.py',
        'src/next2/dependence_runtime.py','src/next2/complement_bank.py','src/next/normalization.py']
    return hashlib.sha256(b''.join((root/p).read_bytes() for p in paths)+json.dumps(settings,sort_keys=True).encode()).hexdigest()
