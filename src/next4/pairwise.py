"""Small same-time pairwise readout fitted on honest inner component OOF only."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

KINDS=('linear','interactions')


def validate(settings):
    if set(settings)!={'version','kind','pairs','seed','temperature','ridge','maxiter'}:
        raise ValueError('Exact pairwise settings required')
    if settings['version']!='honest_same_time_pairs_v1' or settings['kind'] not in KINDS:
        raise ValueError('Unknown pairwise recipe')
    if settings['pairs']!=1048576 or settings['seed']!=20260908 or settings['temperature']!=.1 or settings['ridge']!=.01 or settings['maxiter']!=200:
        raise ValueError('No unregistered pairwise hyperparameters')


def features(scores,kind):
    if kind not in KINDS:raise ValueError('Registered pairwise feature kind required')
    x=np.asarray(scores,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=2 or not np.isfinite(x).all() or ((x<0)|(x>1)).any():
        raise ValueError('Exactly two aligned finite component probabilities required')
    j,c=x[:,0],x[:,1]
    if kind=='linear':return x.copy()
    return np.column_stack([j,c,j*j,c*c,j*c,np.abs(j-c)])


def sample_pairs(target,times,series_ids,count,seed):
    y=np.asarray(target);t=np.asarray(times);ids=np.asarray(series_ids)
    if y.ndim!=1 or t.shape!=y.shape or ids.shape!=y.shape or not np.isin(y,[0,1]).all():
        raise ValueError('Aligned binary training targets/time/group coordinates required')
    if not np.isfinite(t).all() or (t<0).any() or (t!=np.floor(t)).any():raise ValueError('Integral nonnegative time required')
    if type(count) is not int or count<1:raise ValueError('Positive pair count required')
    order=np.argsort(t,kind='stable');cuts=np.r_[0,np.flatnonzero(np.diff(t[order]))+1,len(t)]
    positive=[];negative=[];pair_mass=[];eligible_times=[]
    for start,end in zip(cuts[:-1],cuts[1:]):
        rows=order[start:end];p=rows[y[rows]==1];n=rows[y[rows]==0]
        if len(p) and len(n):
            positive.append(p);negative.append(n);pair_mass.append(len(p)*len(n));eligible_times.append(int(t[rows[0]]))
    if not pair_mass:raise ValueError('No eligible same-time positive-negative pairs')
    mass=np.asarray(pair_mass,dtype=np.int64);rng=np.random.default_rng(seed)
    time_samples=rng.choice(len(mass),size=count,p=mass.astype(float)/mass.sum())
    frequencies=np.bincount(time_samples,minlength=len(mass));left=np.empty(count,dtype=np.int64);right=left.copy();cursor=0
    for k,amount in enumerate(frequencies):
        end=cursor+int(amount)
        left[cursor:end]=positive[k][rng.integers(len(positive[k]),size=amount)]
        right[cursor:end]=negative[k][rng.integers(len(negative[k]),size=amount)]
        cursor=end
    if not np.array_equal(t[left],t[right]) or not np.all(y[left]==1) or not np.all(y[right]==0):
        raise RuntimeError('Pair sampling violated time/target contract')
    if np.any(ids[left]==ids[right]):raise ValueError('Positive and negative pair must be different series at the same time')
    proof={'pairs':count,'seed':seed,'eligible_times':len(mass),'eligible_pair_mass':int(mass.sum()),
        'same_time_pairs':True,'different_series_pairs':True,'training_coordinates_sha256':hashlib.sha256(np.c_[left,right].tobytes()).hexdigest(),
        'time_sampling':'Time sampled in proportion to training positive_count * negative_count; uniform positive and negative within sampled time',
        'objective_scope':'Unbiased Monte Carlo surrogate risk for the official training-time pair-mass weighting; not an unbiased validation score estimate'}
    return left,right,proof


def loss_gradient(weights,differences,anchor,temperature,ridge):
    margins=differences@weights/temperature
    error=weights-anchor
    loss=float(np.logaddexp(0.,-margins).mean()+.5*ridge*(error@error))
    gradient=-(differences.T@expit(-margins))/(len(differences)*temperature)+ridge*error
    return loss,gradient


def raw_readout(design,weights):
    # Column-wise operations give the same summation order in scalar and batch inference.
    result=np.zeros(len(design),dtype=np.float64)
    for j in range(len(weights)):result+=design[:,j]*weights[j]
    return result


@dataclass
class PairwiseModel:
    settings:dict
    weights:np.ndarray
    implementation_sha256:str
    training:dict

    def predict(self,scores):
        validate(self.settings)
        if self.implementation_sha256!=source_hash():raise RuntimeError('Pairwise implementation changed')
        # A fixed monotone map provides bounded outputs without fitting a calibration layer.
        values=expit(raw_readout(features(scores,self.settings['kind']),self.weights)-.5)
        return np.asarray(values,dtype=np.float32)

    def predict_one(self,primary,complement):return float(self.predict(np.array([[primary,complement]]))[0])

    def save(self,path):
        path=Path(path)
        if path.exists():raise FileExistsError(path)
        joblib.dump(self,path,compress=3)

    @classmethod
    def load(cls,path):
        model=joblib.load(path)
        if not isinstance(model,cls) or model.implementation_sha256!=source_hash():raise RuntimeError('Invalid pairwise model identity')
        validate(model.settings);return model


def fit(training_scores,training_target,training_times,training_series,settings):
    validate(settings);x=features(training_scores,settings['kind'])
    left,right,pair_proof=sample_pairs(training_target,training_times,training_series,settings['pairs'],settings['seed'])
    differences=x[left]-x[right];anchor=np.zeros(x.shape[1]);anchor[:2]=[.8,.2]
    bounds=[(0.,4.),(0.,4.)]+[(-4.,4.)]*(x.shape[1]-2)
    initial,_=loss_gradient(anchor,differences,anchor,settings['temperature'],settings['ridge'])
    result=minimize(loss_gradient,anchor,args=(differences,anchor,settings['temperature'],settings['ridge']),method='L-BFGS-B',jac=True,
        bounds=bounds,options={'maxiter':settings['maxiter'],'ftol':1e-12,'gtol':1e-8,'maxls':30})
    if not result.success or not np.isfinite(result.x).all():raise RuntimeError('Pairwise optimizer failed: '+str(result.message))
    detail={'pair_sampling':pair_proof,'training_series':len(np.unique(training_series)),'training_rows':len(x),
        'feature_count':x.shape[1],'new_meta_fits':1,'new_base_fits':0,'initial_pair_loss':initial,
        'final_pair_loss':float(result.fun),'optimizer_iterations':int(result.nit),'optimizer_success':bool(result.success),
        'coefficient_bounds':bounds,'ridge_anchor':anchor.tolist(),'intercept_fitted':False,
        'validation_targets_or_scores_accepted_by_fit_API':False}
    return PairwiseModel(dict(settings),result.x.copy(),source_hash(),detail),detail


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
