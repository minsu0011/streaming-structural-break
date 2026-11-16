"""H-only modulation of a small honest nested pairwise combination."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import joblib
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from src.next4 import pairwise
from src.next4.historical_context import context,implementation_hash as context_hash
from src.next4.artifacts import ROOT,BASE,atomic_json,read_json,digest,utc_now,progress

SETTINGS={'version':'H_pairwise_v1','pairs':1048576,'seed':20260908,'temperature':.1,'ridge':.1,'maxiter':200}


def validate(settings):
    if settings!=SETTINGS:raise ValueError('Exact registered H-only pairwise settings required')


def features(scores,h_context):
    base=pairwise.features(scores,'linear');ctx=np.asarray(h_context,dtype=np.float64)
    if ctx.shape!=(len(base),10) or not np.isfinite(ctx).all():raise ValueError('Ten aligned immutable H-only summaries required')
    return np.column_stack([base,(base[:,0]-base[:,1])[:,None]*ctx])


def row_context(context_ids,values,row_ids):
    ids=np.asarray(context_ids);x=np.asarray(values);rows=np.asarray(row_ids)
    if ids.ndim!=1 or x.shape!=(len(ids),10) or rows.ndim!=1 or np.any(np.diff(ids)<=0):raise ValueError('Unique sorted H-context coordinate table required')
    index=np.searchsorted(ids,rows)
    if np.any(index>=len(ids)) or not np.array_equal(ids[index],rows):raise ValueError('Missing H context coordinate')
    return x[index]


def load_historical_cache(data):
    ids=np.asarray(data.ids,dtype=np.int64);assert np.all(np.diff(ids)>0)
    source=context_hash({'version':'historical_context_v1','kind':'base'})
    identity={'feature_sha256':source,'ids_sha256':hashlib.sha256(ids.tobytes()).hexdigest(),
        'split_sha256':data.guard.split.digest,'seal_lock_sha256':data.guard.lock_sha,
        'raw_sha256':read_json(BASE/'preflight/FROZEN_REFERENCES.json')['raw_sha256']}
    out=ROOT/'data/next4'/('H_context_'+source[:16]);manifest=out/'MANIFEST.json'
    if manifest.exists():
        record=read_json(manifest);assert record['identity']==identity
        for name,sha in record['files_sha256'].items():assert digest(out/name)==sha
        np.testing.assert_array_equal(np.load(out/'ids.npy',allow_pickle=False),ids)
        return ids,np.load(out/'context.npy',mmap_mode='r',allow_pickle=False),record
    out.mkdir(parents=True,exist_ok=False);values=np.empty((len(ids),10),dtype=np.float32)
    for j,sid in enumerate(ids):
        h,_,_=data.get(int(sid));values[j]=context(h)
        if (j+1)%1000==0:
            progress('Building shared immutable H context',str(j+1)+' DEV histories; no targets, tau or online points passed to context function')
            print('NEXT4 H CONTEXT',j+1,flush=True)
    np.save(out/'ids.npy',ids,allow_pickle=False);np.save(out/'context.npy',values,allow_pickle=False)
    record={'status':'PASS','created_utc':utc_now(),'identity':identity,'features':10,'series':len(ids),'new_fits':0,
        'outcome_or_online_values_used':False,'seal_rows':0,'files_sha256':{name:digest(out/name) for name in ['ids.npy','context.npy']}}
    atomic_json(manifest,record,immutable=True);data.guard.check_state()
    return ids,values,record


@dataclass
class ContextPairwiseModel:
    settings:dict
    weights:np.ndarray
    implementation_sha256:str
    training:dict

    def predict(self,scores,h_context):
        validate(self.settings)
        if source_hash()!=self.implementation_sha256:raise RuntimeError('H-context pairwise source changed')
        return np.asarray(expit(pairwise.raw_readout(features(scores,h_context),self.weights)-.5),dtype=np.float32)

    def predict_one(self,j,c,h_context):
        j=float(j);c=float(c);ctx=np.asarray(h_context,dtype=np.float64)
        if not 0<=j<=1 or not 0<=c<=1 or ctx.shape!=(10,) or not np.isfinite(ctx).all():
            raise ValueError('Finite current component probabilities and H context required')
        values=np.r_[j,c,(j-c)*ctx];raw=np.float64(0.)
        for weight,value in zip(self.weights,values):raw+=weight*value
        return float(np.float32(expit(raw-.5)))

    def save(self,path):
        path=Path(path)
        if path.exists():raise FileExistsError(path)
        joblib.dump(self,path,compress=3)

    @classmethod
    def load(cls,path):
        model=joblib.load(path)
        if not isinstance(model,cls) or model.implementation_sha256!=source_hash():raise RuntimeError('Invalid H-context pairwise model')
        validate(model.settings);return model


def fit(scores,h_context,target,times,series_ids,settings):
    validate(settings);x=features(scores,h_context)
    left,right,proof=pairwise.sample_pairs(target,times,series_ids,settings['pairs'],settings['seed'])
    differences=x[left]-x[right];anchor=np.r_[.8,.2,np.zeros(10)];bounds=[(0.,4.),(0.,4.)]+[(-1.,1.)]*10
    initial,_=pairwise.loss_gradient(anchor,differences,anchor,settings['temperature'],settings['ridge'])
    result=minimize(pairwise.loss_gradient,anchor,args=(differences,anchor,settings['temperature'],settings['ridge']),method='L-BFGS-B',jac=True,
        bounds=bounds,options={'maxiter':settings['maxiter'],'ftol':1e-12,'gtol':1e-8,'maxls':30})
    if not result.success or not np.isfinite(result.x).all():raise RuntimeError('H-context optimizer failed: '+str(result.message))
    detail={'pair_sampling':proof,'training_rows':len(x),'training_series':len(np.unique(series_ids)),'feature_count':12,
        'new_meta_fits':1,'new_base_fits':0,'initial_pair_loss':initial,'final_pair_loss':float(result.fun),
        'optimizer_iterations':int(result.nit),'optimizer_success':bool(result.success),'coefficient_bounds':bounds,
        'ridge_anchor':anchor.tolist(),'context_main_effects':False,'only_H_modulates_expert_score_gap':True,
        'validation_targets_or_scores_accepted_by_fit_API':False}
    return ContextPairwiseModel(dict(settings),result.x.copy(),source_hash(),detail),detail


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()+pairwise.source_hash().encode()+context_hash({'version':'historical_context_v1','kind':'base'}).encode()).hexdigest()
