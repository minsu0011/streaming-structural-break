"""A small supervised readout of causal state features, without base-score inputs."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import joblib
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits
from src.next.fitting import weights_from_training
from src.next4 import temporal_state


def loss_gradient(theta,x,y,weights,ridge):
    margin=x@theta[1:]+theta[0];total=weights.sum()
    error=(expit(margin)-y)*weights/total
    loss=float(np.dot(weights,np.logaddexp(0.,margin)-y*margin)/total+.5*ridge*(theta[1:]@theta[1:]))
    gradient=np.r_[error.sum(),x.T@error+ridge*theta[1:]]
    return loss,gradient


@dataclass
class StateReadout:
    theta:np.ndarray
    settings:dict
    feature_sha256:str
    model_source_sha256:str
    training:dict

    def validate(self):
        if self.feature_sha256!=temporal_state.implementation_hash(self.settings) or self.model_source_sha256!=source_hash():
            raise RuntimeError('Temporal readout source changed')

    def predict(self,x):
        x=np.asarray(x,dtype=np.float64)
        if x.ndim!=2 or x.shape[1]!=32 or not np.isfinite(x).all():raise ValueError('Exactly 32 finite causal state features required')
        margin=np.full(len(x),self.theta[0],dtype=np.float64)
        for j in range(32):margin+=x[:,j]*self.theta[j+1]
        return np.asarray(expit(margin),dtype=np.float32)

    def predict_one(self,x):
        x=np.asarray(x,dtype=np.float64)
        if x.shape!=(32,) or not np.isfinite(x).all():raise ValueError('Finite current causal state vector required')
        value=np.float64(self.theta[0])
        for coefficient,point in zip(self.theta[1:],x):value+=coefficient*point
        return float(np.float32(expit(value)))

    def make_state(self,historical):
        self.validate();return temporal_state.make_state(historical,self.settings)

    def save(self,path):
        path=Path(path)
        if path.exists():raise FileExistsError(path)
        joblib.dump(self,path,compress=3)

    @classmethod
    def load(cls,path):
        model=joblib.load(path)
        if not isinstance(model,cls):raise TypeError('Registered temporal readout required')
        model.validate();return model


def fit_base(x,metadata,train,valid,*,guard,registration):
    train=np.asarray(train,dtype=np.int64);valid=np.asarray(valid,dtype=np.int64)
    if len(np.unique(train))!=len(train) or len(np.unique(valid))!=len(valid) or np.intersect1d(train,valid).size:
        raise ValueError('Disjoint unique original group partitions required')
    guard.partition(np.unique(metadata['dataset_id'][train]),np.unique(metadata['dataset_id'][valid]))
    parameters=registration['parameters'];settings=registration['feature_settings']
    if parameters!={'learner':'fixed_state_logistic','weighting':'W3','ridge':.01,'maxiter':200,'frozen_BL80_weight':.9}:
        raise ValueError('Exact fixed temporal-state readout parameters required')
    if registration['feature_sha256']!=temporal_state.implementation_hash(settings):raise RuntimeError('State feature sources changed')
    matrix=np.asarray(x[train],dtype=np.float64);target=np.asarray(metadata['target'][train],dtype=np.float64)
    if matrix.shape!=(len(train),32) or not np.isfinite(matrix).all() or set(np.unique(target))!={0.,1.}:
        raise ValueError('Finite state features and both training classes required')
    weights=weights_from_training(target,metadata['time_online'][train],metadata['dataset_id'][train],'W3')
    probability=float(np.average(target,weights=weights));initial=np.zeros(33);initial[0]=np.log(probability/(1-probability))
    with threadpool_limits(limits=4):
        initial_loss,_=loss_gradient(initial,matrix,target,weights,parameters['ridge'])
        result=minimize(loss_gradient,initial,args=(matrix,target,weights,parameters['ridge']),method='L-BFGS-B',jac=True,
            options={'maxiter':200,'ftol':1e-12,'gtol':1e-8,'maxls':30})
    if not result.success or not np.isfinite(result.x).all():raise RuntimeError('Temporal readout optimizer failed: '+str(result.message))
    detail={'training_rows':len(train),'validation_rows':len(valid),'training_series':len(np.unique(metadata['dataset_id'][train])),
        'validation_series':len(np.unique(metadata['dataset_id'][valid])),'feature_count':32,'readout_parameters':33,
        'initial_training_loss':initial_loss,'final_training_loss':float(result.fun),'optimizer_iterations':int(result.nit),
        'optimizer_success':True,'weighting':'W3','weight_statistics_training_only':True,'new_fits':1,
        'validation_labels_used_in_fit':False,'base_prediction_inputs_used_in_fit':False,
        'transition_parameters_fitted':False,'readout_coefficients_fitted':True,'seal_rows':0}
    model=StateReadout(result.x.copy(),dict(settings),registration['feature_sha256'],source_hash(),detail)
    return model,model.predict(x[valid]),detail


def source_hash():
    root=Path(__file__).resolve().parents[2]
    return hashlib.sha256(Path(__file__).read_bytes()+(root/'src/next/fitting.py').read_bytes()).hexdigest()
