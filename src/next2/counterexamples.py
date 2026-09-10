"""Locked causal AR/ARCH counterexamples; variance multipliers are variances."""
from pathlib import Path
import json
import hashlib
import numpy as np
from numba import njit

ROOT=Path(__file__).resolve().parents[2]


def policy():
    return json.loads((ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json').read_text(encoding='utf-8'))


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def noise(seed,count,df):
    rng=np.random.default_rng(np.random.SeedSequence(seed))
    return rng.normal(size=count) if df is None else rng.standard_t(df,size=count)*np.sqrt((df-2.)/df)


@njit(cache=True)
def recurrence(eps,phi_before,phi_after,alpha_before,alpha_after,factor,mean,change):
    result=np.empty(len(eps),dtype=np.float64)
    y=0.
    q=1.-phi_before*phi_before
    for t in range(len(eps)):
        after=t>=change
        phi=phi_after if after else phi_before
        alpha=alpha_after if after else alpha_before
        target=(1.-phi*phi)*(factor if after else 1.)
        variance=(1.-alpha)*target+alpha*q
        innovation=np.sqrt(max(variance,1e-12))*eps[t]
        y=phi*y+innovation+(1.-phi)*(mean if after else 0.)
        result[t]=y
        q=innovation*innovation
    return result


def generate(scenario,index,*,partition=1,horizon=None,control=False,lock=None):
    lock=policy() if lock is None else lock
    items=lock['scenarios']
    ordinal=next(j for j,s in enumerate(items) if s['id']==scenario)
    spec=items[ordinal]
    burn=lock['burn_in']; nh=lock['historical_length']
    no=lock['online_length'] if horizon is None else int(horizon)
    scheduled=int(lock['tau_cycle'][index%len(lock['tau_cycle'])])
    # Separate H/pre-change/post-change RNG streams preserve prefixes at every
    # requested online horizon, including different Student-t algorithms.
    seed=[lock['seed'],partition,ordinal,index]
    hist=noise(seed+[0],burn+nh,spec['t_df_before'])
    pre=noise(seed+[1],min(scheduled,no),spec['t_df_before'])
    post_df=spec['t_df_before'] if control else spec['t_df_after']
    post=noise(seed+[2],max(no-scheduled,0),post_df)
    eps=np.r_[hist,pre,post]
    null=spec['id'].startswith('N') or spec['id']=='O1' or control
    phi_after=spec['phi_before'] if control else spec['phi_after']
    alpha_after=spec['arch_before'] if control else spec['arch_after']
    factor=1. if control else spec['variance_factor']
    mean=0. if control else spec['mean_shift_stationary_sd']
    values=recurrence(eps,spec['phi_before'],phi_after,spec['arch_before'],alpha_after,
                      factor,mean,burn+nh+scheduled)
    h=values[burn:burn+nh].astype(np.float32)
    o=values[burn+nh:].astype(np.float32)
    if spec['id']=='O1' and not control and scheduled<no:
        o[scheduled]+=20.*(1 if index%2==0 else -1)
    if not np.isfinite(h).all() or not np.isfinite(o).all():
        raise RuntimeError('Non-finite fixed synthetic DGP')
    return h,o,-1 if null else scheduled


@njit(cache=False)
def predict_kernel(kernel,args,points):
    result=np.empty(len(points),dtype=np.float32)
    for j in range(len(points)):
        result[j]=kernel(float(points[j]),args)
    return result


def predict(prepared,h,o):
    state=prepared.new_state(h)
    return predict_kernel(state.kernel,state.call_args,np.asarray(o,dtype=np.float32))
