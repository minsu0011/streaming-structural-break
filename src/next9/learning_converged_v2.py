"""Bounded optimizer continuation, retaining the registered logistic objective."""
import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits
from src.next9.learning import fit_classifier as fit_original
from src.next9.movement_screen import group_weights

def logistic_stationarity(model,x,y,groups):
    if model.estimator is None:return dict(objective=0.,normalized_gradient_max=0.)
    if model.classes!=2 or model.training['method']!='LOGISTIC':raise ValueError('Binary logistic diagnostic only')
    z=model.transform(x);w=group_weights(groups);e=model.estimator
    logits=z@e.coef_[0]+e.intercept_[0];residual=w*(expit(logits)-np.asarray(y))
    objective=float(np.sum(w*(np.logaddexp(0,logits)-np.asarray(y)*logits))+.5*np.sum(e.coef_**2)/e.C)
    gradient=np.r_[z.T@residual+e.coef_[0]/e.C,residual.sum()]/w.sum()
    return dict(objective=objective,normalized_gradient_max=float(np.max(np.abs(gradient))))

def fit_classifier(x,y,ids,groups,scope,*,method='LOGISTIC',classes=2):
    model=fit_original(x,y,ids,groups,scope,method=method,classes=classes)
    return continue_classifier(model,x,y,ids,groups,scope)

def continue_classifier(model,x,y,ids,groups,scope):
    scope.admit_training(ids)
    assert model.training['training_ids_sha256']==scope.provenance()['training_ids_sha256']
    method=model.training['method'];classes=model.classes
    if model.estimator is None or method!='LOGISTIC' or model.training.get('converged',True):return model
    if classes!=2:raise ValueError('Continuation is only preregistered for binary joint M heads')
    before=logistic_stationarity(model,x,y,groups)
    old=dict(model.training)
    model.estimator.set_params(warm_start=True,max_iter=5000)
    with threadpool_limits(limits=1):
        model.estimator.fit(model.transform(x),np.asarray(y).astype(int),sample_weight=group_weights(groups))
    after=logistic_stationarity(model,x,y,groups)
    converged=bool(np.max(model.estimator.n_iter_)<5000)
    if not converged:raise RuntimeError('Single bounded5000-iteration continuation did not converge')
    if after['objective']>before['objective']+1e-7*max(1.,abs(before['objective'])):
        raise RuntimeError('Same-objective continuation increased regularized TRAIN loss')
    model.training.update(estimator_fit_calls=old['estimator_fit_calls']+1,
        iterations=model.estimator.n_iter_.tolist(),converged=True,parameters=model.estimator.get_params(),
        optimizer_continuation=dict(original_iterations=old['iterations'],original_parameters=old['parameters'],
            before=before,after=after,extra_fit_calls=1,additional_iteration_cap=5000,
            same_C_and_group_weights=True,same_causal_inputs_labels_and_standardization=True,
            no_new_scientific_candidate=True,validation_or_outer_scores_used=False))
    return model
