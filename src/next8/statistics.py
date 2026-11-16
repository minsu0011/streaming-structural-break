"""Series-level uncertainty and same-time ranking; no pooled-row authority."""
import hashlib
import numpy as np
from scipy.stats import norm
from numba import njit
from src.scoring.ts_auc import ts_auc


def series_weights(ids):
    _,code,count=np.unique(ids,return_inverse=True,return_counts=True)
    w=1./count[code]
    return w/w.mean()


def series_mean(ids,values):
    keys,code,count=np.unique(ids,return_inverse=True,return_counts=True)
    values=np.asarray(values)
    if values.ndim==1:
        means=np.bincount(code,weights=values)/count
    else:
        means=np.column_stack([np.bincount(code,weights=values[:,j])/count for j in range(values.shape[1])])
    return keys,means


def cluster_difference(ids,reference_loss,candidate_loss,*,draws=2000,seed=8101604):
    keys,delta=series_mean(ids,np.asarray(reference_loss)-np.asarray(candidate_loss))
    n=len(keys);estimate=float(delta.mean())
    if n<2:return {'series':n,'gain':estimate,'ci_low':None,'ci_high':None,'p_normal_cluster':1.,'status':'INSUFFICIENT_SUPPORT'}
    se=float(delta.std(ddof=1)/np.sqrt(n))
    code=int(hashlib.sha256(np.asarray(keys).tobytes()).hexdigest()[:8],16)
    rng=np.random.default_rng(seed^code)
    # Batches bound memory; draw family identical for every comparison with same IDs.
    samples=[]
    for start in range(0,draws,64):
        ix=rng.integers(0,n,size=(min(64,draws-start),n))
        samples.append(delta[ix].mean(axis=1))
    boot=np.concatenate(samples)
    p=float(2*norm.sf(abs(estimate)/se)) if se>0 else (0. if estimate else 1.)
    return {'series':n,'gain':estimate,'ci_low':float(np.quantile(boot,.025)),
            'ci_high':float(np.quantile(boot,.975)),'p_normal_cluster':p,
            'positive_resample_fraction':float(np.mean(boot>0)),'draws':draws,
            'unit':'whole_series','status':'SUPPORTED' if n>=64 else 'INSUFFICIENT_SUPPORT',
            'p_approximation':'two-sided normal approximation using series means; not a randomization p-value'}


def holm(pvalues):
    p=np.asarray(pvalues,dtype=float);order=np.argsort(p,kind='stable')
    adjusted=np.maximum.accumulate(np.minimum(1.,p[order]*(len(p)-np.arange(len(p)))))
    out=np.empty(len(p));out[order]=adjusted
    return out


def binary_losses(y,p):
    y=np.asarray(y);p=np.clip(np.asarray(p),1e-7,1-1e-7)
    return -(y*np.log(p)+(1-y)*np.log1p(-p))


def multiclass_losses(y,p):
    return -np.log(np.clip(np.asarray(p)[np.arange(len(y)),np.asarray(y,dtype=int)],1e-7,1.))


def classification_metrics(y,p,ids):
    y=np.asarray(y,dtype=int);p=np.asarray(p);prediction=np.argmax(p,axis=1)
    weights=series_weights(ids)
    recalls=[];counts=[]
    for k in range(p.shape[1]):
        keep=y==k;counts.append(int(keep.sum()))
        recalls.append(float(np.average(prediction[keep]==k,weights=weights[keep])) if np.any(keep) else None)
    target=np.eye(p.shape[1])[y]
    return {'balanced_accuracy':float(np.mean([x for x in recalls if x is not None])),
            'recall_by_class':recalls,'class_counts':counts,
            'logloss':float(np.average(multiclass_losses(y,p),weights=weights)),
            'Brier':float(np.average(np.sum((p-target)**2,axis=1),weights=weights)),
            'independent_series':int(len(np.unique(ids))),'rows':len(y)}


def regression_metrics(y,p,ids):
    error=np.asarray(p)-np.asarray(y);w=series_weights(ids)
    return {'MAE':float(np.average(np.abs(error),weights=w)),
            'MSE':float(np.average(error**2,weights=w)),
            'independent_series':int(len(np.unique(ids))),'rows':len(y)}


def ranking(y,p,t,ids):
    score,details=ts_auc(y,p,t,return_details=True)
    return {'TS_AUC':float(score),'independent_series':len(np.unique(ids)),
            'pairs':sum(r['weight'] for r in details),'supported_online_times':sum(r['weight']>0 for r in details),
            'positive_series':len(np.unique(np.asarray(ids)[np.asarray(y)==1])),
            'negative_series':len(np.unique(np.asarray(ids)[np.asarray(y)==0]))}


@njit(cache=False)
def _rank_endpoints(yy,pp,tt):
    n=len(yy);wins=np.zeros(n);degree=np.zeros(n)
    total_pairs=0.;total_wins=0.;a=0
    while a<n:
        b=a+1
        while b<n and tt[b]==tt[a]:b+=1
        positive=0
        for k in range(a,b):positive+=int(yy[k])
        negative=b-a-positive
        if positive>0 and negative>0:
            cum_positive=0;cum_negative=0;u=a
            while u<b:
                v=u+1
                while v<b and pp[v]==pp[u]:v+=1
                pos=0
                for k in range(u,v):pos+=int(yy[k])
                neg=v-u-pos
                for k in range(u,v):
                    if yy[k]:wins[k]=cum_negative+.5*neg;degree[k]=negative
                    else:wins[k]=positive-cum_positive-.5*pos;degree[k]=positive
                total_wins+=pos*(cum_negative+.5*neg)
                cum_positive+=pos;cum_negative+=neg;u=v
            total_pairs+=positive*negative
        a=b
    return wins,degree,total_pairs,total_wins


def rank_influence(y,p,t,ids):
    """First-order whole-series influence for the exact pair-weighted TS-AUC.

    Ties contribute half a win. Summing both positive and negative endpoints
    captures shared-series contributions across times. Used for SCREENING
    cluster-normal intervals; final model uncertainty uses full bootstrap.
    """
    y=np.asarray(y,dtype=np.int8);p=np.asarray(p);t=np.asarray(t);ids=np.asarray(ids)
    order=np.lexsort((p,t));yy=y[order];pp=p[order];tt=t[order]
    wins,degree,total_pairs,total_wins=_rank_endpoints(yy,pp,tt)
    keys,code=np.unique(ids[order],return_inverse=True)
    auc=total_wins/total_pairs if total_pairs else .5
    influence=np.bincount(code,weights=(wins-auc*degree)/max(total_pairs,1.))
    se=float(np.sqrt(len(keys)/max(len(keys)-1,1)*np.sum(influence**2)))
    return {'TS_AUC':float(auc),'cluster_se':se,'ci_low':float(max(0.,auc-1.96*se)),
            'ci_high':float(min(1.,auc+1.96*se)),'pairs':int(total_pairs),'series':len(keys),
            'positive_series':len(np.unique(ids[y==1])),'negative_series':len(np.unique(ids[y==0])),
            'interval_method':'First-order same-time U-statistic series-cluster normal; approximate screening interval'},keys,influence


def logistic_contrast(model,design,y,ids,weights,positive_name,negative_name):
    """Direct conditional coefficient difference with cluster sandwich SE."""
    if model.estimator is None or not hasattr(model.estimator,'coef_'):
        return {'difference':0.,'ci_low':None,'ci_high':None,'p_value':1.,'status':'INSUFFICIENT_SUPPORT'}
    z=model.transform(design);z=np.column_stack([np.ones(len(z)),z])
    p=model.predict_proba(design)[:,1];w=np.asarray(weights,dtype=float);w=w/w.mean()
    curvature=w*p*(1-p)
    h=z.T@(z*curvature[:,None]);h.flat[::len(h)+1]+=10.;h[0,0]-=10.
    inv=np.linalg.pinv(h,hermitian=True)
    _,code=np.unique(ids,return_inverse=True);groups=int(code.max())+1
    scores=np.zeros((groups,z.shape[1]))
    np.add.at(scores,code,z*((np.asarray(y)-p)*w)[:,None])
    covariance=inv@(scores.T@scores)@inv*groups/max(groups-1,1)
    contrast=np.zeros(z.shape[1])
    for name,sign in ((positive_name,1.),(negative_name,-1.)):
        if name in model.names:
            j=model.names.index(name);contrast[j+1]=sign/model.scale[j]/4.
    coefficients=np.r_[model.estimator.intercept_[0],model.estimator.coef_[0]]
    difference=float(contrast@coefficients)
    se=float(np.sqrt(max(float(contrast@covariance@contrast),0.)))
    pvalue=float(2*norm.sf(abs(difference)/se)) if se else (0. if difference else 1.)
    return {'difference':difference,'cluster_se':se,'ci_low':difference-1.96*se,
            'ci_high':difference+1.96*se,'p_value':pvalue,'independent_series':groups,
            'status':'SUPPORTED' if groups>=64 else 'INSUFFICIENT_SUPPORT',
            'method':'Training-fitted logit coefficient contrast; regularized Hessian and whole-series sandwich, approximate normal interval'}
