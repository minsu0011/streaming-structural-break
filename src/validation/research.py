"""Development-only CV building blocks. No seal/reduced target reads."""
import numpy as np
from src.features.streaming import FEATURE_NAMES,feature_names
from src.features.config import CONFIG
from src.models.learners import fit_candidate
from src.scoring.ts_auc import ts_auc
from src.validation.splits import assert_development_ids

def sample_frame(frame,policy):
    if policy in ('S0','S3','S3_sqrt'): return frame
    if policy in ('S1','S2'): return frame.loc[frame[f'sample_{policy}']==1]
    raise ValueError('Sampling policy is not frozen')

def contribution_weights(target,time,policy='S3'):
    """Each row is weighted by its opposite-class count at the same TRAIN time.

    Thus total weight per t is 2*n_positive*n_negative, matching the TS-AUC
    contribution structure. No validation prevalence or outcomes are consulted.
    """
    y=np.asarray(target,dtype=np.uint8);t=np.asarray(time,dtype=np.int64)
    positive=np.bincount(t,weights=y);negative=np.bincount(t,weights=1-y,minlength=len(positive))
    weight=np.where(y==1,negative[t],positive[t])
    if policy=='S3_sqrt':weight=np.sqrt(weight)
    elif policy!='S3':raise ValueError('Unfrozen weighting family')
    if weight.sum()<=0:raise ValueError('No training time has positive-negative pairs')
    return weight/weight.mean()

def fit_fold(folds,validation_fold,name,families,policy,*,split=None,params=None,config=CONFIG):
    import pandas as pd
    validation=folds[validation_fold]
    training=pd.concat([sample_frame(frame,policy) for fold,frame in enumerate(folds) if fold!=validation_fold],ignore_index=True)
    train_ids=set(training.dataset_id);val_ids=set(validation.dataset_id)
    assert_development_ids(train_ids|val_ids,split)
    if train_ids&val_ids: raise RuntimeError('Series overlap between train/validation')
    if split is not None and {split.groups[i] for i in train_ids}&{split.groups[i] for i in val_ids}:
        raise RuntimeError('Historical duplicate group overlaps training and validation')
    cross_order=name=='M7_cross_order_equal'
    if cross_order:
        if policy!='S3':raise ValueError('Fixed equal blend requires the frozen S3 weighting')
        from src.features.cross_order import feature_names as cross_names
        names=cross_names(config)
    else:names=feature_names(config)
    matrix=training.loc[:,names].to_numpy(dtype=np.float32)
    # Constant removal is learned separately on TRAINING folds only.
    excluded=tuple(np.flatnonzero(np.ptp(matrix,axis=0)<=1e-12))
    weights=contribution_weights(training.target,training.time_online,policy) if policy.startswith('S3') else None
    if cross_order:
        from src.models.cross_order import fit_cross_order
        model=fit_cross_order(matrix,training.target.to_numpy(),families=families,sample_weight=weights,params=params,config=config)
    else:model=fit_candidate(name,matrix,training.target.to_numpy(),families=families,sample_weight=weights,params=params,config=config,excluded_columns=excluded)
    # The official socket converts every emitted score to float32. Score the same
    # quantized values locally so tie behavior matches deployment for all models.
    prediction=np.asarray(model.predict(validation.loc[:,names].to_numpy(dtype=np.float32)),dtype=np.float32)
    score=ts_auc(validation.target.to_numpy(),prediction,validation.time_online.to_numpy())
    return model,prediction,score

def summarize(scores):
    values=np.asarray(scores,dtype=float)
    if len(values)!=5 or not np.isfinite(values).all(): raise ValueError('All five folds must complete')
    return {'fold_scores':values.tolist(),'mean':float(values.mean()),'median':float(np.median(values)),
        'std':float(values.std()),'worst_fold':float(values.min()),'best_fold':float(values.max())}
