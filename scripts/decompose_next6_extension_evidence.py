"""Fixed-score attribution: predictive state information versus relationship excess."""
from pathlib import Path
import sys,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next6_extension/numba_cache')
import numpy as np,joblib
from numba import njit
from src.next6_extension.common import *
from src.next6.data import arrays
from src.next6.nodes import FAMILIES,NAMES
from src.scoring.ts_auc import ts_auc
from src.next5.bootstrap import PreparedAUC

@njit(cache=True)
def per_node(states,initial,parents,tables,own_tables):
    ring=initial.copy();out=np.zeros((len(states),16,3),dtype=np.float64)
    for t in range(len(states)):
        for j in range(16):
            s=int(states[t,j]);own=int(ring[(t-1)%16,j]);ctx=own;good=s<3 and own<3
            for k in range(2):
                i,lag=parents[j,k]
                if i<0:continue
                p=int(ring[(t-lag)%16,i]);ctx+=p*3**(k+1);good=good and p<3
            if good:
                full=np.log(tables[1,j,ctx,s]/tables[0,j,ctx,s]);marginal=np.log(own_tables[1,j,own,s]/own_tables[0,j,own,s])
                out[t,j,0]=full;out[t,j,1]=marginal;out[t,j,2]=full-marginal
        ring[t%16]=states[t]
    return out

def main():
    data,meta,v=arrays();metrics=[];attribution=[];boot=[]
    for f in range(5):
        with activity(f'evidence_decomposition_fold{f}','BOTTLENECK_ATTRIBUTION'):
            directory=ROOT/f'artifacts/next6/fold_{f}/graphs/outer_full';g=joblib.load(directory/'graph.joblib');rows=np.load(directory/'rows.npy');features=np.load(directory/'features.npy',mmap_mode='r');assert np.all(meta['fold'][rows]==f)
            names=('full_transition_logratio','own_state_logratio','conditional_parent_excess')
            full=features[:,13].astype(float);excess=features[:,23].astype(float)-features[:,22].astype(float);signals=np.column_stack((full,full-excess,excess))
            y=meta['target'][rows];t=meta['time_online'][rows];ids=meta['dataset_id'][rows];unique,codes=np.unique(ids,return_inverse=True);n=len(unique)
            rng=np.random.default_rng(2026091302+f);w=np.vstack((np.ones((1,n),dtype=np.int32),rng.multinomial(n,np.full(n,1/n),size=1000)))
            bb=[]
            for j,name in enumerate(names):
                score=float(ts_auc(y,signals[:,j],t));p=PreparedAUC.create(y,signals[:,j],t,codes,n);z=p.evaluate(w);bb.append(z)
                lo,hi=np.quantile(z[1:],[.025,.975]);metrics.append({'fold':f,'signal':name,'TS_AUC':score,'ci_2_5':lo,'ci_97_5':hi,'scope':'univariate diagnostic score, not a trained detector or new candidate','sign_fixed':True})
            boot.append(np.column_stack(bb))
            par=np.full((16,2,2),-1,dtype=np.int64)
            for j in range(16):
                for k,(i,lagindex) in enumerate(g.parents[j]):par[j,k]=(i,(1,2,4,8,16)[lagindex])
            values={}
            for sid in unique:
                i=data.id_to_index[int(sid)];a,b=data.series['online_offsets'][i:i+2];tau=int(data.series['tau'][i]);r=np.arange(a,b);states=v['states'][r]
                out=per_node(states,v['H_states'][i],par,g.tables,g.own_tables);rel=np.arange(b-a)-tau
                masks={'no_break':np.ones(b-a,dtype=bool) if tau<0 else np.zeros(b-a,dtype=bool),
                    'pre':(rel>=-32)&(rel<0) if tau>=0 else np.zeros(b-a,dtype=bool),
                    'onset':(rel>=0)&(rel<=7) if tau>=0 else np.zeros(b-a,dtype=bool),
                    'earlyPost':(rel>=8)&(rel<=31) if tau>=0 else np.zeros(b-a,dtype=bool),
                    'establishedPost':(rel>=32)&(rel<=127) if tau>=0 else np.zeros(b-a,dtype=bool),
                    'laterPost':rel>=128 if tau>=0 else np.zeros(b-a,dtype=bool)}
                for phase,mask in masks.items():
                    if mask.any():values.setdefault(phase,[]).append((out[mask].mean(axis=0),np.abs(out[mask]).mean(axis=0)))
            for phase,parts in values.items():
                z=np.array([p[0] for p in parts]);az=np.array([p[1] for p in parts]);n=len(z);bw=rng.multinomial(n,np.full(n,1/n),size=1000)
                flat=z.reshape(n,-1);lo,hi=np.quantile(bw@flat/n,[.025,.975],axis=0);lo=lo.reshape(16,3);hi=hi.reshape(16,3)
                for node in range(16):
                    for j,name in enumerate(names):attribution.append({'fold':f,'phase':phase,'node':node,'node_name':NAMES[node],'family':FAMILIES[node],'signal':name,'independent_series':n,'mean_log_score':z[:,node,j].mean(),'mean_absolute_log_score':az[:,node,j].mean(),'ci_2_5':lo[node,j],'ci_97_5':hi[node,j],'parents':len(g.parents[node]),'equal_series_weight':True})
    csv_write('FROZEN_EVIDENCE_DECOMPOSITION_AUC.csv',metrics);csv_write('FROZEN_EVIDENCE_NODE_ATTRIBUTION.csv',attribution)
    b=np.mean(np.stack(boot,axis=-1),axis=-1);delta=b[:,0]-b[:,1];lo,hi=np.quantile(delta[1:],[.025,.975])
    save('FROZEN_EVIDENCE_DECOMPOSITION.json',{'status':'COMPLETE','mean_TS_AUC':dict(zip(names,b[0].tolist())),'full_minus_own_TS_AUC':float(delta[0]),'paired_CI':[float(lo),float(hi)],'identity':'full = own + conditional parent excess; zero-parent nodes have exact zero excess','interpretation':'No graph tuning or learned weights. Diagnostic discrimination is distinct from M3 head performance. All outer results were previously released; this is adaptive explanatory analysis.'})
    data.guard.check_state()
if __name__=='__main__':main()
