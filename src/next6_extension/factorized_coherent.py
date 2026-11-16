"""Coherent sparse support repair of the frozen factorized interaction graph.

Back off a whole own-previous-state stratum across BOTH phases and ALL parent
states if any of its six phase/parent cells has fewer8 supporting series. This
keeps every returned null table free of phase-parent-current interaction.
Original factorized.py and its results remain available for audit.
"""
import hashlib
import numpy as np
from src.next6_extension.factorized import Graph,fit_pair as original_pair,evaluate_pair
from src.next6_extension.fresh_state import FAMILIES
from src.next6.common import ids_hash

def null_interaction_error(null):
    maximum=0.
    for own in range(3):
        for a in range(2):
            for b in range(2):
                odds=null[:,own,a,b]*null[:,own,a+1,b+1]/(null[:,own,a,b+1]*null[:,own,a+1,b]);maximum=max(maximum,abs(float(np.log(odds[0])-np.log(odds[1]))))
    return maximum

def fit_pair(current,previous,phase,ids,source,target):
    full,null,support,receipt=original_pair(current,previous,phase,ids,source,target);raw_support=support.copy();blocked=np.any(support<8,axis=(0,2))
    good=(phase>=0)&(current[:,target]<3)&(previous[:,target]<3)&(previous[:,source]<3);cur=current[good,target];own=previous[good,target];label=phase[good];sid=ids[good]
    owncount=np.full((2,3,3),1.5)
    for p in (0,1):
        mask=label==p;unique,code,n=np.unique(sid[mask],return_inverse=True,return_counts=True);w=64./n[code];np.add.at(owncount[p],(own[mask],cur[mask]),w)
    ownprob=owncount/owncount.sum(axis=-1,keepdims=True)
    for ownstate in np.flatnonzero(blocked):
        full[:,ownstate]=ownprob[:,ownstate,None,:];null[:,ownstate]=ownprob[:,ownstate,None,:];support[:,ownstate]=0
    error=null_interaction_error(null)
    if error>1e-8:raise RuntimeError(f'Coherent null interaction invariant failed: {error}')
    receipt=receipt|{'whole_ownstate_backoff':np.flatnonzero(blocked).tolist(),'raw_unique_series_support':raw_support.tolist(),'returned_support':'effective eligible support; zero for whole-stratum backoff','returned_null_interaction_error':error}
    return full,null,support,receipt

class CoherentGraph(Graph):
    def validate(self):
        super().validate()
        for j in range(8):
            if null_interaction_error(self.null[:,j])>1e-8:raise ValueError('Null contains an unintended phase interaction')

def discover_and_fit(current,previous,phase,ids,train_ids,group_map):
    allowed=set(map(int,train_ids));assert set(map(int,ids))<=allowed
    order=sorted(allowed,key=lambda sid:(hashlib.sha256(('NEXT6_DISCOVERY:'+str(group_map[sid])).encode()).hexdigest(),sid));gh={}
    for j,sid in enumerate(order):gh.setdefault(group_map[sid],j%2)
    halves=np.array([gh[group_map[int(sid)]] for sid in ids]);records=[];eligible=[]
    for target in range(8):
        for source in range(8):
            if source==target or FAMILIES[source]==FAMILIES[target]:continue
            gains=[];okay=True
            for half in (0,1):
                tr=halves!=half;va=~tr;table=fit_pair(current[tr],previous[tr],phase[tr],ids[tr],source,target)
                evaluated=evaluate_pair(table,current[va],previous[va],phase[va],ids[va],source,target);train_support=[len(np.unique(ids[tr&(phase==p)])) for p in (0,1)]
                valid=all(e['series']>=32 for e in evaluated) and min(train_support)>=32;gain=float(np.mean([e['mean_gain'] for e in evaluated])) if valid else None;frac=float(np.mean([e['positive_fraction'] for e in evaluated])) if valid else None
                accepted=valid and gain>=.001 and frac>=.55;okay=okay and accepted;gains.append(gain)
                records.append({'source':source,'target':target,'raw_lag':16,'half':half,'balanced_full_vs_null_gain':gain,'balanced_positive_series_fraction':frac,'phase0_train_series':train_support[0],'phase1_train_series':train_support[1],
                    'phase0_valid_series':evaluated[0]['series'],'phase1_valid_series':evaluated[1]['series'],'eligible_half':accepted,'whole_ownstate_backoff':table[3]['whole_ownstate_backoff'],'returned_null_interaction_error':table[3]['returned_null_interaction_error']})
            if okay:eligible.append((min(gains),target,source))
    parents=np.full(8,-1,dtype=np.int64);selected=[]
    for gain,target,source in sorted(eligible,key=lambda e:(-e[0],e[1],e[2])):
        if parents[target]>=0:continue
        parents[target]=source;selected.append({'source':source,'target':target,'raw_lag':16,'minimum_half_gain':gain,'shared_raw':True,'economic_causality':False})
    full=np.full((2,8,3,3,3),1/3);null=full.copy();support=np.zeros((2,8,3,3),dtype=np.int32);receipts=[]
    for e in selected:
        j=e['target'];a,b,c,receipt=fit_pair(current,previous,phase,ids,e['source'],j);full[:,j]=a;null[:,j]=b;support[:,j]=c;receipts.append(receipt)
    graph=CoherentGraph(parents,full,null,support,selected,{'training_ids':sorted(allowed),'training_ids_sha256':ids_hash(allowed),'clean_blocks':int(np.sum(phase>=0)),
        'selection':'same fixed two internal grouped heldout gains; coherent whole-ownstate support fallback','outer_labels_selection':False,'fit_receipts':receipts,
        'repair':'phase-dependent context fallback replaced with whole ownstate backoff in both phases; no new thresholds, weights, nodes or lags','null_invariant':'phase-parent-current interaction exactly absent for each own previous state'})
    graph.validate();return graph,records
