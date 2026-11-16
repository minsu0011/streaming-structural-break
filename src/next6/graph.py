"""Train-group-only sparse predictive relations and factorized transitions."""
from dataclasses import dataclass
import hashlib
import numpy as np
from numba import njit
from scipy.stats import ttest_1samp
from threadpoolctl import threadpool_limits
from src.next6.nodes import LAGS,NAMES,FAMILIES
from src.next6.common import ids_hash,checkpoint

REL_NAMES=('normal_prediction_abs_residual','post_minus_normal_regression_logscore',
    'residual_ewma8','residual_ewma32','recent_relationship_weakening',
    'large_residual_node_fraction','lagged_parent_confirmation_fraction','missing_expected_response_fraction',
    'confirmed_persistence_log_duration','normal_recovery_log_duration','unconfirmed_shock','confirmed_persistent_state')
STATE_NAMES=('normal_transition_surprise','post_minus_normal_transition_logscore','max_normal_transition_surprise',
    'rare_transition_fraction','transition_difference_ewma8','transition_difference_ewma32',
    'concurrent_changed_node_fraction','missing_state_fraction','transition_surprise_ewma8',
    'transition_surprise_ewma32','parent_vs_own_normal_logscore','parent_vs_own_post_logscore')

def design(temporal,time):
    # Five actual lag blocks (80), known current online time, intercept.
    return np.column_stack((np.asarray(temporal[:,16:96],dtype=np.float64),np.log1p(time)/8.,np.ones(len(time))))

def owncols(j):return sorted(set([j+16*k for k in range(5)]+[4,7,80,81]))

def ridge(g,xy,cols):
    gram=g[np.ix_(cols,cols)].copy();penalty=np.full(len(cols),.05)
    if 81 in cols:penalty[cols.index(81)]=1e-8
    return np.linalg.solve(gram+np.diag(penalty),xy[cols])

def weights(sid):
    _,ix,count=np.unique(sid,return_inverse=True,return_counts=True)
    w=1./count[ix];return w/w.sum()

def pack_sample(data,metadata,arrays,train_ids):
    data.guard.admit(train_ids,purpose='NEXT6 graph training')
    indices=[]
    for sid in sorted(train_ids):
        k=data.id_to_index[int(sid)];a,b=data.series['online_offsets'][k:k+2]
        # Temporal features retain all intermediate points even when supervised rows are subsampled.
        for phase in (0,1):
            take=np.arange(a,b,8,dtype=np.int64);take=take[metadata['target'][take]==phase][:128]
            indices.extend(take.tolist())
    rows=np.asarray(indices,dtype=np.int64)
    temporal=np.asarray(arrays['tabular'][rows,101:])
    return {'rows':rows,'X':design(temporal,metadata['time_online'][rows]),
        'Y':np.asarray(arrays['nodes'][rows],dtype=np.float64),'phase':metadata['target'][rows],
        'ids':metadata['dataset_id'][rows]}

def mechanically_related(i,j):
    return 'MECHANICAL_DERIVED' if FAMILIES[i]==FAMILIES[j] or {FAMILIES[i],FAMILIES[j]}<={'energy','variance','conditional_energy'} or {i,j}<={0,1,4,8,9,10,11} else 'SHARED_RAW_INPUT_POSSIBLE'

def discover(sample,group_map):
    X,Y,phase,ids=(sample[k] for k in ('X','Y','phase','ids'))
    unique=sorted(set(map(int,ids)),key=lambda sid:(hashlib.sha256(('NEXT6_DISCOVERY:'+str(group_map[sid])).encode()).hexdigest(),sid))
    halves={sid:i%2 for i,sid in enumerate(unique)}
    # RP duplicate groups are singleton, but enforce grouping rather than assume it.
    group_halves={}
    for sid in unique:group_halves.setdefault(group_map[sid],halves[sid])
    assignment=np.array([group_halves[group_map[int(sid)]] for sid in ids])
    menu=[(i,j,k) for j in range(16) for i in range(16) if i!=j for k in range(5)]
    records={edge:[] for edge in menu}
    with threadpool_limits(limits=2):
        for label in (0,1):
            for half in (0,1):
                tr=(phase==label)&(assignment!=half);va=(phase==label)&(assignment==half)
                ti=np.flatnonzero(tr);vi=np.flatnonzero(va)
                support=len(np.unique(ids[vi]));train_support=len(np.unique(ids[ti]))
                if support<32 or train_support<32:
                    for edge in menu:records[edge].append({'phase':label,'half':half,'support':support,'gain':None,'positive_fraction':None,'coefficient':None,'p_diagnostic':None})
                    continue
                a=X[ti];y=Y[ti];w=weights(ids[ti]);g=a.T@(a*w[:,None]);xy=a.T@(y*w[:,None])
                vx=X[vi];vy=Y[vi];vs=ids[vi];_,codes,counts=np.unique(vs,return_inverse=True,return_counts=True)
                bases={}
                for j in range(16):
                    cols=owncols(j);coef=ridge(g,xy[:,j],cols);err=vy[:,j]-vx[:,cols]@coef
                    bases[j]=(cols,err,np.bincount(codes,weights=err*err)/counts)
                for i,j,k in menu:
                    basecols,berr,bmse=bases[j];column=i+16*k
                    if column in basecols:
                        gain=0.;frac=0.;coeflast=0.;pval=1.
                    else:
                        cols=basecols+[column];coef=ridge(g,xy[:,j],cols);err=vy[:,j]-vx[:,cols]@coef
                        cmse=np.bincount(codes,weights=err*err)/counts
                        changes=(bmse-cmse)/np.maximum(bmse,.01)
                        gain=float(np.mean(changes));frac=float(np.mean(changes>0));coeflast=float(coef[-1])
                        pval=float(ttest_1samp(changes,0,alternative='greater').pvalue) if np.std(changes)>0 else 1.
                        if not np.isfinite(pval):pval=1.
                    records[(i,j,k)].append({'phase':label,'half':half,'support':support,'gain':gain,'positive_fraction':frac,'coefficient':coeflast,'p_diagnostic':pval})
                checkpoint('inner edge scan',f'label={label} half={half}; training_series={train_support} validation_series={support}')
    rows=[];eligible=[]
    for (i,j,k),tests in records.items():
        best=-np.inf;chosen=None
        for phaseval in (0,1):
            v=[r for r in tests if r['phase']==phaseval]
            ok=len(v)==2 and all(r['gain'] is not None and r['support']>=32 and r['gain']>=.005 and r['positive_fraction']>=.55 for r in v)
            ok=ok and v[0]['coefficient']*v[1]['coefficient']>0
            if ok and min(r['gain'] for r in v)>best:best=min(r['gain'] for r in v);chosen=phaseval
        for r in tests:rows.append({'source':i,'target':j,'lag':LAGS[k],'source_name':NAMES[i],'target_name':NAMES[j],
            'mechanical':mechanically_related(i,j),'eligible_phase':chosen,**r})
        if chosen is not None:eligible.append((best,j,i,k,chosen))
    parents=[[] for _ in range(16)];selected=[]
    for gain,j,i,k,phaseval in sorted(eligible,key=lambda r:(-r[0],r[1],r[2],r[3])):
        if len(selected)>=24:break
        if len(parents[j])>=2 or any(p[0]==i for p in parents[j]):continue
        parents[j].append((i,k));selected.append({'source':i,'target':j,'lag':LAGS[k],'lag_index':k,
            'selection_gain':gain,'selection_phase':phaseval,'mechanical':mechanically_related(i,j)})
    return selected,rows

def state_contexts(data,arrays,rows,parents):
    current=np.asarray(arrays['states'][rows]);past=np.empty((len(rows),16,5),dtype=np.int8)
    offsets=data.series['online_offsets'];ix=np.searchsorted(offsets[1:],rows,side='right');time=rows-offsets[ix]
    for k,lag in enumerate(LAGS):
        valid=time>=lag
        past[valid,:,k]=arrays['states'][rows[valid]-lag]
        past[~valid,:,k]=arrays['H_states'][ix[~valid],16+time[~valid]-lag]
    contexts=np.ones((len(rows),16),dtype=np.int16)*-1
    own=past[:,:,0].copy()
    for j in range(16):
        c=own[:,j].astype(np.int16);good=(c<3)&(current[:,j]<3)
        for m,(i,k) in enumerate(parents[j]):
            v=past[:,i,k];good&=v<3;c=c+v*(3**(m+1))
        contexts[good,j]=c[good]
    return current,own,contexts

@dataclass
class Graph:
    parents:list
    selected:list
    coef:np.ndarray
    own_coef:np.ndarray
    variance:np.ndarray
    tables:np.ndarray
    own_tables:np.ndarray
    context_support:np.ndarray
    provenance:dict

def fit_graph(data,metadata,arrays,train_ids,*,topology='selected',selected=None):
    train_ids=set(map(int,train_ids));sample=pack_sample(data,metadata,arrays,train_ids)
    if selected is None and topology=='selected':selected,edge_rows=discover(sample,data.guard.split.groups)
    else:edge_rows=[];selected=[] if selected is None else selected
    parents=[[] for _ in range(16)]
    if topology=='random':
        rng=np.random.default_rng(int(ids_hash(train_ids)[:8],16)^20260913)
        random=[]
        for j in range(16):
            count=sum(e['target']==j for e in selected)
            sources=rng.choice([i for i in range(16) if i!=j],count,replace=False)
            for i in sources:
                k=int(rng.integers(5));random.append({'source':int(i),'target':j,'lag':LAGS[k],'lag_index':k,'mechanical':mechanically_related(int(i),j)})
        selected=random
    if topology=='own':selected=[]
    for e in selected:parents[e['target']].append((e['source'],e['lag_index']))
    X,Y,label,ids=(sample[k] for k in ('X','Y','phase','ids'))
    coef=np.zeros((2,16,82));oc=np.zeros_like(coef);var=np.ones((2,16));tables=np.zeros((2,16,27,3));own_tables=np.zeros((2,16,3,3));support=np.zeros((2,16,27),dtype=np.int32)
    cur,own,ctx=state_contexts(data,arrays,sample['rows'],parents)
    with threadpool_limits(limits=2):
        for phase in (0,1):
            mask=label==phase
            if len(np.unique(ids[mask]))<16:mask=np.ones(len(ids),dtype=bool)
            x=X[mask];y=Y[mask];w=weights(ids[mask]);g=x.T@(x*w[:,None]);xy=x.T@(y*w[:,None])
            for j in range(16):
                cols=owncols(j);oc[phase,j,cols]=ridge(g,xy[:,j],cols)
                cols=sorted(set(cols+[i+16*k for i,k in parents[j]]));coef[phase,j,cols]=ridge(g,xy[:,j],cols)
                err=y[:,j]-x@coef[phase,j];var[phase,j]=max(float(np.sum(w*err*err)),.01)
                ca=cur[mask,j];ow=own[mask,j];cx=ctx[mask,j];sid=ids[mask]
                # Each series supplies total pseudo-count 64; effective shrinkage is explicit.
                cw=w*len(np.unique(sid))*64
                owncount=np.full((3,3),.5);good=(ca<3)&(ow<3)
                np.add.at(owncount,(ow[good],ca[good]),cw[good]);op=owncount/owncount.sum(axis=1,keepdims=True)
                own_tables[phase,j]=op
                for context in range(27):
                    match=(cx==context)&(ca<3);support[phase,j,context]=len(np.unique(sid[match]))
                    counts=np.full(3,.5)
                    np.add.at(counts,ca[match],cw[match])
                    tables[phase,j,context]=counts/counts.sum() if support[phase,j,context]>=8 else op[context%3]
    provenance={'training_ids':sorted(train_ids),'training_ids_sha256':ids_hash(train_ids),'topology':topology,
        'edge_discovery_ids_subset_of_training':True,'sample_rows':len(X),'cross_edges':len(selected),
        'outer_validation_labels_used':False,'fixed_smoothing':.5,'probability_calibrated':False}
    graph=Graph(parents,selected,coef,oc,var,tables,own_tables,support,provenance)
    return graph,edge_rows

@njit(cache=False)
def graph_replay(temporal,states,initial_states,coef,variance,tables,own_tables,parent_array,summary):
    n=len(states);result=np.empty((n,24),dtype=np.float32);ring=initial_states.copy()
    for t in range(n):
        predictors=np.empty(82)
        predictors[:80]=temporal[t,16:96];predictors[80]=np.log1p(summary[0])/8.;predictors[81]=1.
        phase_loss=np.zeros(2);absres=large=normal_nll=post_nll=max_nll=rare=valid=ownnorm=ownpost=0.
        confirmations=expected=0.;confirmed_nodes=0
        for j in range(16):
            current=temporal[t,j]
            for phase in range(2):
                pred=0.
                for k in range(82):pred+=coef[phase,j,k]*predictors[k]
                residual=(current-pred)/np.sqrt(variance[phase,j]);phase_loss[phase]+=.5*min(residual*residual,100.)+.5*np.log(variance[phase,j])
                if phase==0:absres+=min(abs(residual),10.);large+=abs(residual)>2.
            s=states[t,j];own=ring[(int(summary[0])-1)%16,j];context=int(own);good=s<3 and own<3
            nodeconfirmed=False
            for m in range(2):
                i,k=parent_array[j,m]
                if i<0:continue
                lag=(1,2,4,8,16)[k];past=ring[(int(summary[0])-lag)%16,i]
                context+=int(past)*(3**(m+1));good=good and past<3
                if past<3 and past!=1:
                    expected+=1.
                    sign=1 if coef[0,j,i+k*16]>=0 else -1
                    if s<3 and s!=1 and (int(s)-1)==sign*(int(past)-1):confirmations+=1.;nodeconfirmed=True
            confirmed_nodes+=nodeconfirmed
            if good:
                p0=max(tables[0,j,context,s],1e-8);p1=max(tables[1,j,context,s],1e-8)
                v=-np.log(p0);normal_nll+=v;post_nll-=np.log(p1);max_nll=max(max_nll,v);rare+=p0<.1;valid+=1
                ownnorm-=np.log(max(own_tables[0,j,own,s],1e-8));ownpost-=np.log(max(own_tables[1,j,own,s],1e-8))
        absres/=16.;large/=16.;diff=(phase_loss[0]-phase_loss[1])/16.
        for k,half in enumerate((8.,32.)):
            a=1.-np.exp(-np.log(2.)/half);summary[1+k]=(1-a)*summary[1+k]+a*absres
        conf=confirmations/max(expected,1.);miss=(expected-confirmations)/max(expected,1.)
        if confirmed_nodes>=2:summary[3]=min(summary[3]+1,4096);summary[4]=0
        else:summary[3]=0;summary[4]=min(summary[4]+1,4096)
        persistent=1. if summary[3]>=2 else 0.;shock=1. if large>=.125 and not persistent else 0.
        result[t,:12]=np.array([absres,diff,summary[1],summary[2],summary[1]-summary[2],large,conf,miss,
            np.log1p(summary[3]),np.log1p(summary[4]),shock,persistent])
        den=max(valid,1.);normal_nll/=den;post_nll/=den;tdiff=normal_nll-post_nll
        for k,half in enumerate((8.,32.)):
            a=1.-np.exp(-np.log(2.)/half)
            summary[5+k]=(1-a)*summary[5+k]+a*tdiff;summary[7+k]=(1-a)*summary[7+k]+a*normal_nll
        changed=0.
        for j in range(16):changed+=states[t,j]<3 and states[t,j]!=1
        result[t,12:]=np.array([normal_nll,tdiff,max_nll,rare/den,summary[5],summary[6],changed/16.,1-valid/16.,summary[7],summary[8],ownnorm/den-normal_nll,ownpost/den-post_nll])
        ring[int(summary[0])%16]=states[t];summary[0]+=1
    return result,ring

class GraphState:
    def __init__(self,graph,initial_states):
        self.graph=graph;self.ring=np.asarray(initial_states,dtype=np.int8).copy();self.summary=np.zeros(9)
        self.parents=np.full((16,2,2),-1,dtype=np.int64)
        for j,parents in enumerate(graph.parents):
            for m,p in enumerate(parents):self.parents[j,m]=p
    def replay(self,temporal,states):
        out,self.ring=graph_replay(np.asarray(temporal,dtype=np.float32),np.asarray(states,dtype=np.int8),self.ring,
            self.graph.coef,self.graph.variance,self.graph.tables,self.graph.own_tables,self.parents,self.summary)
        return out
    def update(self,temporal,state):return self.replay(np.asarray(temporal)[None,:],np.asarray(state)[None,:])[0]
