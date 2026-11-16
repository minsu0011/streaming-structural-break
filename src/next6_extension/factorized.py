"""Phase-interaction graphs with a marginal-preserving loglinear null.

No economic causal interpretation. Internal held-out proper scores decide edges;
the full and no-interaction tables are both evaluated at inference.
"""
from dataclasses import dataclass
import hashlib
import numpy as np
from src.next6_extension.fresh_state import FAMILIES
from src.next6.common import ids_hash

def ipf_null(counts):
    c=np.asarray(counts,dtype=float)
    if c.shape!=(2,3,3) or np.any(c<=0) or not np.isfinite(c).all():raise ValueError('Positive phase,parent,current counts required')
    wanted=(c.sum(axis=2),c.sum(axis=1),c.sum(axis=0));x=np.ones_like(c)*c.sum()/18.
    error=np.inf
    for iteration in range(1000):
        x*= (wanted[0]/x.sum(axis=2))[:,:,None]
        x*= (wanted[1]/x.sum(axis=1))[:,None,:]
        x*= (wanted[2]/x.sum(axis=0))[None,:,:]
        error=max(float(np.max(np.abs(x.sum(axis=axis)-target)/np.maximum(target,1e-12))) for axis,target in zip((2,1,0),wanted))
        if error<=1e-10:return x,iteration+1,error
    raise RuntimeError(f'IPF failed convergence: {error}')

def fit_pair(current,previous,phase,ids,source,target):
    good=(phase>=0)&(current[:,target]<3)&(previous[:,target]<3)&(previous[:,source]<3)
    cur=current[good,target];own=previous[good,target];parent=previous[good,source];label=phase[good];sid=ids[good]
    counts=np.full((2,3,3,3),.5);support=np.zeros((2,3,3),dtype=np.int32)
    for p in (0,1):
        mask=label==p;unique,code,n=np.unique(sid[mask],return_inverse=True,return_counts=True);w=64./n[code]
        np.add.at(counts[p],(own[mask],parent[mask],cur[mask]),w)
        for a in range(3):
            for b in range(3):support[p,a,b]=len(np.unique(sid[mask&(own==a)&(parent==b)]))
    fitted=np.empty_like(counts);iterations=[];errors=[]
    for ownstate in range(3):
        fitted[:,ownstate],n,e=ipf_null(counts[:,ownstate]);iterations.append(n);errors.append(e)
    full=counts/counts.sum(axis=-1,keepdims=True);null=fitted/fitted.sum(axis=-1,keepdims=True)
    owncount=counts.sum(axis=2);ownprob=owncount/owncount.sum(axis=-1,keepdims=True)
    for p in (0,1):
        for a in range(3):
            for b in range(3):
                if support[p,a,b]<8:full[p,a,b]=ownprob[p,a];null[p,a,b]=ownprob[p,a]
    return full,null,support,{'max_IPF_iterations':max(iterations),'max_IPF_relative_marginal_error':max(errors)}

def evaluate_pair(tables,current,previous,phase,ids,source,target):
    full,null,support,_=tables;records=[]
    for p in (0,1):
        good=(phase==p)&(current[:,target]<3)&(previous[:,target]<3)&(previous[:,source]<3)
        y=current[good,target];own=previous[good,target];parent=previous[good,source];sid=ids[good]
        unique,code,count=np.unique(sid,return_inverse=True,return_counts=True)
        if not len(unique):records.append({'phase':p,'series':0,'mean_gain':None,'positive_fraction':None,'values':np.empty(0),'ids':unique});continue
        delta=np.log(full[p,own,parent,y])-np.log(null[p,own,parent,y]);by=np.bincount(code,weights=delta)/count
        records.append({'phase':p,'series':len(unique),'blocks':len(y),'mean_gain':float(by.mean()),'positive_fraction':float((by>0).mean()),'values':by,'ids':unique,
            'supported_context_fraction':float(np.mean(support[p,own,parent]>=8))})
    return records

@dataclass
class Graph:
    parents:np.ndarray
    full:np.ndarray
    null:np.ndarray
    support:np.ndarray
    selected:list
    provenance:dict
    def validate(self):
        if self.parents.shape!=(8,) or np.any(self.parents>=8) or np.any(self.parents<-1) or any(int(v)==j for j,v in enumerate(self.parents)):raise ValueError('Invalid sparse topology')
        for a in (self.full,self.null):
            if a.shape!=(2,8,3,3,3) or not np.isfinite(a).all() or np.any(a<=0) or not np.allclose(a.sum(axis=-1),1,atol=1e-12):raise ValueError('Invalid conditional probabilities')
        if self.support.shape!=(2,8,3,3):raise ValueError('Invalid support')

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
                tr=halves!=half;va=~tr
                table=fit_pair(current[tr],previous[tr],phase[tr],ids[tr],source,target)
                evaluated=evaluate_pair(table,current[va],previous[va],phase[va],ids[va],source,target)
                train_support=[len(np.unique(ids[tr&(phase==p)])) for p in (0,1)]
                valid=all(e['series']>=32 for e in evaluated) and min(train_support)>=32
                gain=float(np.mean([e['mean_gain'] for e in evaluated])) if valid else None
                frac=float(np.mean([e['positive_fraction'] for e in evaluated])) if valid else None
                accepted=valid and gain>=.001 and frac>=.55;okay=okay and accepted;gains.append(gain)
                records.append({'source':source,'target':target,'raw_lag':16,'half':half,'balanced_full_vs_null_gain':gain,'balanced_positive_series_fraction':frac,'phase0_train_series':train_support[0],'phase1_train_series':train_support[1],
                    'phase0_valid_series':evaluated[0]['series'],'phase1_valid_series':evaluated[1]['series'],'eligible_half':accepted,**table[3]})
            if okay:eligible.append((min(gains),target,source))
    parents=np.full(8,-1,dtype=np.int64);selected=[]
    for gain,target,source in sorted(eligible,key=lambda e:(-e[0],e[1],e[2])):
        if parents[target]>=0:continue
        parents[target]=source;selected.append({'source':source,'target':target,'raw_lag':16,'minimum_half_gain':gain,'shared_raw':True,'economic_causality':False})
    full=np.full((2,8,3,3,3),1/3);null=full.copy();support=np.zeros((2,8,3,3),dtype=np.int32);receipts=[]
    for e in selected:
        j=e['target'];a,b,c,receipt=fit_pair(current,previous,phase,ids,e['source'],j);full[:,j]=a;null[:,j]=b;support[:,j]=c;receipts.append(receipt)
    graph=Graph(parents,full,null,support,selected,{'training_ids':sorted(allowed),'training_ids_sha256':ids_hash(allowed),'clean_blocks':int(np.sum(phase>=0)),
        'selection':'two internal grouped heldouts full-vs-marginal-preserving-null log score','outer_labels_selection':False,'fit_receipts':receipts})
    graph.validate();return graph,records
