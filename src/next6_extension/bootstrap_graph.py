"""Whole-series weighted resampling of the fixed NEXT6 discovery algorithm.

The original internal hash halves are held fixed. Zero-weight groups provide no
support. Replicates quantify training sampling sensitivity, not unseen-test gain.
"""
import hashlib
import numpy as np
from threadpoolctl import threadpool_limits
from src.next6.graph import owncols,ridge,weights,mechanically_related
from src.next6.nodes import LAGS

def prepare(sample,group_map):
    ids=sample['ids'];unique=sorted(set(map(int,ids)),key=lambda sid:(hashlib.sha256(('NEXT6_DISCOVERY:'+str(group_map[sid])).encode()).hexdigest(),sid))
    group_halves={}
    for i,sid in enumerate(unique):group_halves.setdefault(group_map[sid],i%2)
    groups=sorted(set(group_map[int(s)] for s in ids));lookup={g:i for i,g in enumerate(groups)}
    codes=np.array([lookup[group_map[int(s)]] for s in ids]);half=np.array([group_halves[group_map[int(s)]] for s in ids])
    packed=[]
    for phase in (0,1):
        for h in (0,1):
            ix=np.flatnonzero((sample['phase']==phase)&(half==h));x=np.ascontiguousarray(sample['X'][ix]);y=np.ascontiguousarray(sample['Y'][ix]);sid=ids[ix]
            u,code,count=np.unique(sid,return_inverse=True,return_counts=True)
            packed.append({'x':x,'y':y,'ids':sid,'unique':u,'code':code,'count':count,'group':codes[ix],'unique_group':np.array([lookup[group_map[int(s)]] for s in u])})
    return packed,np.array([group_halves[g] for g in groups]),groups

def discovery(packed,multiplicity):
    # Phase, validation half, target, source, lag; controls already in own model yield zero.
    gains=np.zeros((2,2,16,16,5));fractions=np.zeros_like(gains);coefficients=np.zeros_like(gains)
    supports=np.zeros((2,2),dtype=int)
    with threadpool_limits(limits=2):
        for phase in (0,1):
            for h in (0,1):
                tr=packed[2*phase+1-h];va=packed[2*phase+h]
                tm=multiplicity[tr['unique_group']];vm=multiplicity[va['unique_group']]
                supports[phase,h]=np.count_nonzero(vm)
                if supports[phase,h]<32 or np.count_nonzero(tm)<32:continue
                w=tm[tr['code']]/tr['count'][tr['code']];w=w/w.sum()
                x=tr['x'];y=tr['y'];g=x.T@(x*w[:,None]);xy=x.T@(y*w[:,None]);vx=va['x'];vy=va['y'];codes=va['code'];counts=va['count']
                for j in range(16):
                    cols=owncols(j);b=ridge(g,xy[:,j],cols);berr=vy[:,j]-vx[:,cols]@b
                    bmse=np.bincount(codes,weights=berr*berr,minlength=len(counts))/counts
                    for i in range(16):
                        if i==j:continue
                        for k in range(5):
                            c=i+16*k
                            if c in cols:continue
                            cc=cols+[c];b1=ridge(g,xy[:,j],cc);err=vy[:,j]-vx[:,cc]@b1
                            mse=np.bincount(codes,weights=err*err,minlength=len(counts))/counts
                            delta=(bmse-mse)/np.maximum(bmse,.01)
                            gains[phase,h,j,i,k]=np.dot(vm,delta)/vm.sum()
                            fractions[phase,h,j,i,k]=np.dot(vm,delta>0)/vm.sum()
                            coefficients[phase,h,j,i,k]=b1[-1]
    eligible=[]
    for j in range(16):
        for i in range(16):
            if i==j:continue
            for k in range(5):
                best=-np.inf;chosen=None
                for phase in (0,1):
                    ok=np.all(supports[phase]>=32) and np.all(gains[phase,:,j,i,k]>=.005) and np.all(fractions[phase,:,j,i,k]>=.55) and np.prod(coefficients[phase,:,j,i,k])>0
                    v=float(gains[phase,:,j,i,k].min())
                    if ok and v>best:best=v;chosen=phase
                if chosen is not None:eligible.append((best,j,i,k,chosen))
    parents=[[] for _ in range(16)];selected=[]
    for gain,j,i,k,phase in sorted(eligible,key=lambda z:(-z[0],z[1],z[2],z[3])):
        if len(selected)>=24:break
        if len(parents[j])>=2 or i in parents[j]:continue
        parents[j].append(i);selected.append({'source':i,'target':j,'lag':LAGS[k],'lag_index':k,'selection_phase':phase,'selection_gain':gain,'mechanical':mechanically_related(i,j)})
    return selected,{'supports':supports.tolist(),'eligible_edges':len(eligible)}
