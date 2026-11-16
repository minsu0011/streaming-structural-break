"""Known state-DGP controls for interaction-graph sensitivity, not competition data."""
import numpy as np

def generated(groups,blocks,seed,coupling,occupancy,kind):
    rng=np.random.default_rng(np.random.SeedSequence(seed));prob=np.array([1/3]*3 if occupancy=='uniform' else [.1,.8,.1])
    current=rng.choice(3,size=(groups,blocks,8),p=prob).astype(np.int8);initial=rng.choice(3,size=(groups,8),p=prob).astype(np.int8)
    sourcepast=np.concatenate((initial[:,None,0],current[:,:-1,0]),axis=1);phase=np.broadcast_to((np.arange(blocks)>=blocks//2).astype(np.int8),(groups,blocks))
    choose=rng.random((groups,blocks))<coupling
    if kind in ('changed_link','stable_link'):
        desired=np.where(phase==0,sourcepast,2-sourcepast) if kind=='changed_link' else sourcepast
        current[:,:,4]=np.where(choose,desired,current[:,:,4])
    elif kind=='marginal_shift':
        post=rng.choice(3,size=(groups,blocks),p=[.7,.2,.1]);current[:,:,4]=np.where(phase==1,post,current[:,:,4])
    elif kind!='null':raise ValueError(kind)
    previous=np.concatenate((initial[:,None,:],current[:,:-1,:]),axis=1)
    return current.reshape(-1,8),previous.reshape(-1,8),phase.ravel(),np.repeat(np.arange(groups,dtype=np.int32)+1000000,blocks)
