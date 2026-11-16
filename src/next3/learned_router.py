"""Tiny train-only gates fitted to honestly nested base predictions."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.tree import DecisionTreeClassifier
from src.next.fitting import weights_from_training
from src.next3.router import blend_with_weight


@dataclass
class LogisticGate:
    center: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    training_detail: dict

    def weights(self, features):
        z = np.clip((np.asarray(features,dtype=np.float64)-self.center)/self.scale,-8.,8.)
        x = np.column_stack([np.ones(len(z)),z])
        return .3*expit(x@self.coefficients)

    def predict(self, features):
        return blend_with_weight(features[:,0],features[:,1],self.weights(features))


def logistic_objective(coefficients, x, primary, complement, target, weight, ridge=.01):
    g=expit(x@coefficients)
    raw=primary+.3*g*(complement-primary)
    probability=np.clip(raw,1e-7,1.-1e-7)
    norm=weight.sum()
    loss=-np.sum(weight*(target*np.log(probability)+(1-target)*np.log1p(-probability)))/norm
    loss+=.5*ridge*np.dot(coefficients[1:],coefficients[1:])
    active=(raw>1e-7)&(raw<1.-1e-7)
    response=weight*(probability-target)/(probability*(1.-probability))*.3*g*(1.-g)*(complement-primary)*active/norm
    gradient=x.T@response
    gradient[1:]+=ridge*coefficients[1:]
    return float(loss),gradient


def fit_logistic(features, metadata):
    values=np.asarray(features,dtype=np.float64)
    target=np.asarray(metadata['target'],dtype=np.float64)
    weight=weights_from_training(target,metadata['time_online'],metadata['dataset_id'],'W3')
    center=values.mean(axis=0);scale=np.maximum(values.std(axis=0),1e-6)
    z=np.clip((values-center)/scale,-8.,8.)
    x=np.column_stack([np.ones(len(z)),z])
    initial=np.zeros(x.shape[1]);initial[0]=np.log(2.)
    result=minimize(logistic_objective,initial,args=(x,values[:,0],values[:,1],target,weight),
        method='L-BFGS-B',jac=True,options={'maxiter':100,'ftol':1e-10,'gtol':1e-6,'maxls':30})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError('Registered logistic optimizer failed: '+str(result.message))
    return LogisticGate(center,scale,result.x,{'solver':'L-BFGS-B','iterations':result.nit,
        'function_evaluations':result.nfev,'objective':float(result.fun),'converged':bool(result.success),
        'ridge':.01,'maxiter':100,'training_rows':len(target),'feature_count':values.shape[1]})


@dataclass
class TreeGate:
    estimator: object
    training_detail: dict

    def weights(self, features):
        classes=list(self.estimator.classes_)
        probability=self.estimator.predict_proba(np.asarray(features,dtype=np.float32))[:,classes.index(1)] if 1 in classes else np.zeros(len(features))
        # Exact small menu, round half upward to avoid platform tie ambiguity.
        return np.floor(.3*probability*10.+.5)/10.

    def predict(self, features):
        return blend_with_weight(features[:,0],features[:,1],self.weights(features))


def fit_tree(features, metadata):
    values=np.asarray(features,dtype=np.float32)
    target=np.asarray(metadata['target'],dtype=np.uint8)
    primary=np.clip(values[:,0],1e-7,1-1e-7)
    complement=np.clip(values[:,1],1e-7,1-1e-7)
    primary_loss=-(target*np.log(primary)+(1-target)*np.log1p(-primary))
    complement_loss=-(target*np.log(complement)+(1-target)*np.log1p(-complement))
    # This target is base forecast utility on inner OOF, not a mechanism label.
    winner=(complement_loss<primary_loss).astype(np.uint8)
    importance=np.abs(primary_loss-complement_loss)
    pair_weight=weights_from_training(target,metadata['time_online'],metadata['dataset_id'],'W3')
    weight=pair_weight*np.maximum(importance,1e-8)
    estimator=DecisionTreeClassifier(max_depth=2,min_samples_leaf=.05,random_state=0)
    estimator.fit(values,winner,sample_weight=weight)
    return TreeGate(estimator,{'max_depth':2,'min_samples_leaf_fraction':.05,'random_state':0,
        'node_count':estimator.tree_.node_count,'training_rows':len(target),'feature_count':values.shape[1],
        'target':'C205 inner-OOF log loss strictly lower than J102; never mechanism category'})


def fit_gate(kind, features, metadata):
    if not np.isfinite(features).all() or np.asarray(features).shape[1]!=6:
        raise ValueError('Six finite causal gate inputs required')
    if kind=='R2_LOGISTIC_GATE':return fit_logistic(features,metadata)
    if kind=='R3_DEPTH2_GATE':return fit_tree(features,metadata)
    raise ValueError('Unregistered learned gate')
