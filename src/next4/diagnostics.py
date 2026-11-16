"""Evaluation-only scopes and descriptive diversity; never learner inputs."""
import numpy as np
from scipy.stats import rankdata
from scripts.forensic_next_candidate import rank_within_time
from scripts.forensic_next2_candidate import masks_for
from src.next4.artifacts import ROOT,read_json,digest


def scale_directions(data,series):
    rule=read_json(ROOT/'ROBUSTNESS_SPLIT_LOCK.json')['mechanism'];direction={};errors=[]
    for row in series[series.signature=='SCALE_DOMINANT'].itertuples():
        h,o,tau=data.get(int(row.Index))
        if tau!=row.tau or tau<0:raise ValueError('Diagnostic tau differs from frozen signature')
        pre=np.r_[np.asarray(h,dtype=float),np.asarray(o[:tau],dtype=float)][-rule['window']:]
        post=np.asarray(o[tau:tau+rule['window']],dtype=float)
        if min(len(pre),len(post))<rule['minimum_each_side']:raise ValueError('Insufficient fixed-window mechanism support')
        effect=float(np.log(max(float(post.std()),1e-8)/max(float(pre.std()),1e-8)))
        error=abs(abs(effect)-row.scale_effect)
        if error>1e-12:raise ValueError('Directional SCALE reconstruction differs from frozen magnitude')
        errors.append(error);direction[int(row.Index)]='decrease' if effect<0 else 'increase' if effect>0 else 'equal'
    return direction,{'direction_counts':{kind:sum(v==kind for v in direction.values()) for kind in ('decrease','increase','equal')},
        'maximum_magnitude_reconstruction_error':max(errors,default=0.),
        'definition':'Observed raw-window SD direction within frozen SCALE_DOMINANT signature; not latent conditional-variance parameters',
        'evaluation_only':True,'per_series_direction_export':False}


def evaluation_masks(meta,target,times,directions):
    masks=masks_for(meta,target,times)
    for kind in ('decrease','increase'):
        direction=np.array([directions.get(int(sid),'other')==kind for sid in meta.index])
        masks['mechanism_observed_scale_'+kind]=direction|(meta.tau.to_numpy()<0)
    return masks


def correlation(a,b):
    if len(a)<2 or np.std(a)==0 or np.std(b)==0:return None
    return float(np.corrcoef(a,b)[0,1])


def diversity(expert,reference,times,target,expert_ranks=None,reference_ranks=None):
    a=np.asarray(expert);b=np.asarray(reference)
    ra=rank_within_time(a,times) if expert_ranks is None else expert_ranks
    rb=rank_within_time(b,times) if reference_ranks is None else reference_ranks
    result={'pearson':correlation(a,b),'spearman_pooled':correlation(rankdata(a),rankdata(b)),
        'within_time_spearman':correlation(ra,rb),'mean_absolute_rank_disagreement':float(np.abs(ra-rb).mean()),
        'mean_absolute_score_disagreement':float(np.abs(a-b).mean())}
    for percent in (1,5):
        left=ra>=1-percent/100;right=rb>=1-percent/100;negative=target==0
        result['top'+str(percent)+'_jaccard']=float((left&right).sum()/max((left|right).sum(),1))
        result['top'+str(percent)+'_disagreement']=float((left!=right).mean())
        result['top'+str(percent)+'_negative_point_disagreement']=float((left!=right)[negative].mean()) if negative.any() else None
    return result


def source_hash():
    paths=['src/next4/diagnostics.py','scripts/forensic_next2_candidate.py','scripts/forensic_next_candidate.py',
        'src/next/bootstrap.py','src/validation/group_bootstrap.py']
    return {p:digest(ROOT/p) for p in paths}
