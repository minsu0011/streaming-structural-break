"""Small deterministic estimators with train-scoped fitting and causal schemas."""
from dataclasses import dataclass
import hashlib
import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from src.next8.data import FitScope
from src.next8.features import UNKNOWN

FORBIDDEN_INPUT_TOKENS=('future_target','future_value','tau','true_break_age','true_phase',
                        'validation_label','target_orientation_label','total_online_length','future_suffix')


@dataclass(frozen=True)
class Design:
    values: np.ndarray
    names: tuple
    ids: np.ndarray
    scope_kind: str='OBSERVABLE_PREFIX'

    def validate(self):
        if self.scope_kind!='OBSERVABLE_PREFIX':raise ValueError('Only observable prefix inputs')
        x=np.asarray(self.values)
        if x.ndim!=2 or len(x)!=len(self.ids) or x.shape[1]!=len(self.names) or not np.isfinite(x).all():
            raise ValueError('Finite causal design and exact schema required')
        if any(any(word in str(n).lower() for word in FORBIDDEN_INPUT_TOKENS) for n in self.names):
            raise ValueError('Unobservable input name')
        return x


@dataclass
class ScaledEstimator:
    center: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray
    names: tuple
    training: dict
    estimator: object=None
    classes: int=0
    constant_probabilities: np.ndarray=None

    def transform(self,design):
        x=design.validate()
        if tuple(design.names)!=self.names:raise ValueError('Changed inference design schema')
        return (x-self.center)/self.scale

    def predict(self,design):
        z=self.transform(design)
        if self.estimator is None:return z@self.coefficients+self.intercept
        with threadpool_limits(limits=1):return self.estimator.predict(z)

    def predict_proba(self,design):
        z=self.transform(design)
        if self.constant_probabilities is not None:
            return np.tile(self.constant_probabilities,(len(z),1))
        with threadpool_limits(limits=1):
            if self.estimator is not None:
                partial=self.estimator.predict_proba(z)
                out=np.full((len(z),self.classes),1e-8)
                out[:,self.estimator.classes_.astype(int)]=partial
                return out/out.sum(axis=1,keepdims=True)
        p=expit(z@self.coefficients+self.intercept)
        return np.column_stack([1-p,p])


def _prepare(design,y,scope,weight):
    x=design.validate().astype(np.float64)
    scope.admit_training(np.unique(design.ids))
    y=np.asarray(y)
    if y.shape[0]!=len(x) or not np.isfinite(y).all():raise ValueError('Finite training target required')
    w=np.ones(len(x)) if weight is None else np.asarray(weight,dtype=float)
    if w.shape!=(len(x),) or np.any(w<0) or not np.isfinite(w).all() or w.sum()<=0:
        raise ValueError('Valid training weights required')
    w=w/w.mean();center=np.average(x,axis=0,weights=w)
    scale=np.sqrt(np.average((x-center)**2,axis=0,weights=w));scale=np.maximum(scale,1e-6)
    return (x-center)/scale,y,w,center,scale


def fit_ridge(design,y,scope,weight=None,alpha=10.):
    z,y,w,center,scale=_prepare(design,y,scope,weight)
    intercept=np.average(y,axis=0,weights=w)
    a=z.T@(z*w[:,None]);a.flat[::len(a)+1]+=alpha
    b=z.T@((y-intercept)*w if y.ndim==1 else (y-intercept)*w[:,None])
    with threadpool_limits(limits=1):coef=np.linalg.solve(a,b)
    return ScaledEstimator(center,scale,coef,intercept,tuple(design.names),
                           {**scope.provenance(),'method':'ridge','alpha':alpha})


def fit_logistic(design,y,scope,weight=None,classes=2):
    z,y,w,center,scale=_prepare(design,y,scope,weight)
    y=y.astype(int)
    if np.any(y<0) or np.any(y>=classes):raise ValueError('Invalid target class')
    support=np.bincount(y,weights=w,minlength=classes)
    model=ScaledEstimator(center,scale,np.zeros(z.shape[1]),0.,tuple(design.names),
        {**scope.provenance(),'method':'logistic','C':.1,'classes':classes},classes=classes)
    if len(np.unique(y))<2:
        model.constant_probabilities=(support+.5)/(support.sum()+.5*classes)
        return model
    estimator=LogisticRegression(C=.1,max_iter=250,tol=1e-6,solver='lbfgs')
    with threadpool_limits(limits=1):estimator.fit(z,y,sample_weight=w)
    model.estimator=estimator
    model.training['iterations']=estimator.n_iter_.tolist()
    model.training['converged']=bool(np.max(estimator.n_iter_)<250)
    return model


def fit_tree(design,y,scope,weight=None,regression=False):
    z,y,w,center,scale=_prepare(design,y,scope,weight)
    params=dict(max_depth=3,max_iter=100,learning_rate=.05,min_samples_leaf=128,
                l2_regularization=10.,early_stopping=False,random_state=8101604)
    estimator=(HistGradientBoostingRegressor if regression else HistGradientBoostingClassifier)(**params)
    with threadpool_limits(limits=1):estimator.fit(z,y,sample_weight=w)
    return ScaledEstimator(center,scale,np.zeros(z.shape[1]),0.,tuple(design.names),
        {**scope.provenance(),'method':'histgb_depth3',**params},estimator=estimator,classes=0 if regression else int(np.max(y))+1)


def one_hot(states):
    states=np.asarray(states,dtype=np.int8)
    if np.any(states<0) or np.any(states>UNKNOWN):raise ValueError('Unknown state coding')
    return (states[:,None]==np.arange(4)[None,:]).astype(np.float32)


def orientation_design(feature,previous_state,context,ids,method='O2',cross_state=None):
    """Current feature transforms and states are distinct named coordinates."""
    f=np.asarray(feature,dtype=np.float32).reshape(-1)/4.
    c=np.asarray(context,dtype=np.float32)
    if c.ndim==1:c=c[:,None]
    values=[np.ones(len(f)),*c.T]
    names=['intercept_context']+[f'observed_context_{i}' for i in range(c.shape[1])]
    if method!='STATE_ONLY':
        values.append(f);names.append('F_linear')
        if method!='O0':
            values.extend([np.maximum(f,0),np.maximum(-f,0),f*f])
            names.extend(['F_positive','F_negative','F_square'])
    if method in ('O2','O3','O4','STATE_ONLY'):
        own=one_hot(previous_state)
        for k in (0,2,3):values.append(own[:,k]);names.append(f'own_state_{k}')
        if method!='STATE_ONLY':
            for k in (0,2):values.append(own[:,k]*f);names.append(f'own_state_{k}_times_F')
        if cross_state is not None:
            cross=one_hot(cross_state)
            for k in (0,2,3):values.append(cross[:,k]);names.append(f'cross_state_{k}')
            if method!='STATE_ONLY':
                for k in (0,2):values.append(cross[:,k]*f);names.append(f'cross_state_{k}_times_F')
    return Design(np.column_stack(values).astype(np.float32),tuple(names),np.asarray(ids))


def feature_contribution(model,design,unknown_mask):
    """Training-fixed β(state)*F plus fixed nonlinearity, excluding state-only terms.

    Returned values are score contributions, never observed confirmations.
    UNKNOWN removes this whole variable branch, not the stable/base score.
    """
    if model.estimator is None or not hasattr(model.estimator,'coef_'):
        return np.zeros(len(design.ids))
    model.transform(design)  # Exact schema and finite-observation check.
    x=np.asarray(design.values)
    coef=model.estimator.coef_[0]
    selected=np.asarray([n.startswith('F_') or n.endswith('_times_F') for n in model.names])
    out=x[:,selected]@(coef[selected]/model.scale[selected])
    return np.where(np.asarray(unknown_mask,dtype=bool),0.,out)


def estimate_target_orientation(model,feature_value,state,cross_state=None):
    """Derivative w.r.t. F, from TRAIN coefficients; no row target argument."""
    if model.estimator is None or not hasattr(model.estimator,'coef_'):
        return 0.
    b=dict(zip(model.names,model.estimator.coef_[0]/model.scale))
    f=float(feature_value)/4.
    slope=b.get('F_linear',0.)
    slope+=b.get('F_positive',0.) if f>0 else -b.get('F_negative',0.) if f<0 else 0.
    slope+=2*f*b.get('F_square',0.)
    slope+=b.get(f'own_state_{state}_times_F',0.)
    if cross_state is not None:slope+=b.get(f'cross_state_{cross_state}_times_F',0.)
    return float(slope/4.) if state!=UNKNOWN else 0.


def forecast_record(value,horizon,observed_time):
    return {'predicted_feature':float(value),'horizon':int(horizon),'issued_at_observed_time':int(observed_time),
            'kind':'FORECAST','observed_event':False,'confirmed_break':False}


def crossfit_join(expected_ids,parts):
    """Reject meta predictions whose selection or fit included their own group."""
    expected=np.asarray(expected_ids)
    out=np.full(len(expected),np.nan)
    covered=np.zeros(len(expected),dtype=bool)
    for part in parts:
        positions=np.asarray(part['positions'],dtype=int)
        validation=set(map(int,expected[positions]));training=set(map(int,part['training_ids']))
        selection=set(map(int,part['selection_ids']))
        if validation & (training|selection):raise ValueError('Meta prediction selection/fit overlap')
        if np.any(covered[positions]) or len(np.unique(positions))!=len(positions):raise ValueError('Duplicate OOF rows')
        out[positions]=part['prediction'];covered[positions]=True
    if not np.all(covered) or not np.isfinite(out).all():raise ValueError('Incomplete cross-fitted predictions')
    return out


def select_training_only(evidence,scope,ids,kind):
    if kind not in ('feature_classification','state','graph'):raise ValueError('Explicit selection kind')
    scope.admit_training(ids)
    return {'kind':kind,'evidence':evidence,'provenance':scope.provenance()}
