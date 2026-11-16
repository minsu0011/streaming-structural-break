"""Conditional paired-seed metrics with fixed calibration and exact float64 ties."""
import hashlib
import numpy as np
from src.next5.artifacts import ROOT,BASE,read_json,atomic_json,digest,utc_now
from src.next5.bootstrap import PreparedAUC,paired_summary
from src.scoring.ts_auc import ts_auc


def load_cohort(root,key):
    record=read_json(root/key/'RESULTS.json');path=ROOT/record['prediction_file']
    if record['status']!='COMPLETE' or not path.resolve().is_relative_to(root.resolve()) or digest(path)!=record['prediction_file_sha256']:
        raise RuntimeError('Immutable synthetic cohort bytes required')
    with np.load(path,allow_pickle=False) as source:arrays={key:source[key] for key in source.files}
    return arrays,record


def score_inputs(tau,seed_index,horizon):
    times=np.broadcast_to(np.arange(horizon,dtype=np.int32),(len(tau),horizon))
    target=((np.asarray(tau)[:,None]>=0)&(times>=np.asarray(tau)[:,None])).astype(np.uint8)
    return target.ravel(),times.ravel(),np.repeat(seed_index,horizon).astype(np.int32)


def score(prediction,tau,seed_index,horizon):
    target,times,_=score_inputs(tau,seed_index,horizon)
    return float(ts_auc(target,np.asarray(prediction)[:,:horizon].ravel(),times))


def score_draws(prediction,tau,seed_index,horizon,counts):
    target,times,code=score_inputs(tau,seed_index,horizon)
    if not target.any():return np.full(len(counts),.5,dtype=np.float64)
    return PreparedAUC.create(target,np.asarray(prediction)[:,:horizon].ravel(),times,code,counts.shape[1]).evaluate(counts)


def reference_score_draws(prediction,tau,seed_index,horizon,counts,*,scope,fold,scenario,role,record,counts_sha,execution_sha):
    out=BASE/'synthetic/reference_bootstrap'/scope/('fold_'+str(fold));out.mkdir(parents=True,exist_ok=True)
    name=scenario+'_'+role+'_'+str(horizon);path=out/(name+'.npy');receipt=out/(name+'.json')
    identity={'scope':scope,'fold':fold,'scenario':scenario,'role':role,'horizon':horizon,
        'prediction_file_sha256':record['prediction_file_sha256'],'counts_sha256':counts_sha,'execution_sha256':execution_sha}
    if receipt.exists():
        previous=read_json(receipt)
        if previous['identity']!=identity or digest(path)!=previous['sha256']:raise RuntimeError('Cached paired reference scores changed')
        return np.load(path,allow_pickle=False)
    if path.exists():raise RuntimeError('Partial reference bootstrap requires preserved recovery review')
    values=score_draws(prediction,tau,seed_index,horizon,counts);np.save(path,values,allow_pickle=False)
    atomic_json(receipt,{'created_utc':utc_now(),'identity':identity,'sha256':digest(path),'new_fits':0},immutable=True)
    return values


def mean_draws(values,counts):
    values=np.asarray(values,dtype=np.float64)
    if values.ndim!=1 or values.shape[0]!=counts.shape[1] or not np.isfinite(values).all():raise ValueError('One finite value per paired generating seed required')
    return (counts@values)/counts.sum(axis=1)


def paired_mean_difference(candidate,reference,counts):
    new=mean_draws(candidate,counts);old=mean_draws(reference,counts)
    return paired_summary(old[:,None],new[:,None])


def event_by_seed(prediction,tau,seed_index,cutoff,event):
    prediction=np.asarray(prediction);tau=np.asarray(tau);ids=np.asarray(seed_index);n=int(ids.max())+1
    alarm=prediction>cutoff;crossed=alarm.any(axis=1);first=np.where(crossed,alarm.argmax(axis=1),-1)
    if event=='null_eventual':eligible=tau<0;value=crossed
    elif event=='prebreak':eligible=tau>=0;value=crossed&(first<tau)
    elif event=='postbreak_detection':eligible=tau>=0;value=crossed&(first>=tau)
    else:raise ValueError('Fixed alarm event required')
    numerator=np.bincount(ids,weights=eligible&value,minlength=n);denominator=np.bincount(ids,weights=eligible,minlength=n)
    return numerator,denominator


def event_draws(prediction,tau,seed_index,cutoff,event,counts):
    numerator,denominator=event_by_seed(prediction,tau,seed_index,cutoff,event)
    bottom=counts@denominator
    if not np.all(bottom>0):return None
    return (counts@numerator)/bottom
