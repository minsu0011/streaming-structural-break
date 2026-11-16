"""Finite raw influence, disjoint-block confirmation and observable state paths.

This returns evidence and operational states, never a calibrated break probability.
The original NEXT6 classifier is unchanged. All learned references come from H.
"""
import numpy as np
from numba import njit
from src.next.normalization import ecdf_uniform

NAMES=('signed_rank','rank_energy','rank_tail','rank_asymmetry','rank_dependence_lag1','rank_dependence_lag4','rank_energy_dependence_lag1','rank_energy_dependence_lag4')
KOREAN=('위치·부호 순위','크기 순위','꼬리 빈도','비대칭 순위','1시차 순위 의존','4시차 순위 의존','1시차 크기 의존','4시차 크기 의존')
FAMILIES=('location','scale','tail','asymmetry','dependence','dependence','volatility_dependence','volatility_dependence')
STATES=('WARMUP','NORMAL','SHOCK','UNCONFIRMED','PERSISTENT','RECOVERING','INSUFFICIENT_REFERENCE')
STATE_KO=('준비','정상 범위','순간 충격','추가 확인 대기','지속 관측 확인','회복 확인','기준 자료 부족')
COLUMNS=('observable_state','persistent_evidence','current_block_severity','raw_shock','block_complete','completed_blocks','dominant_node','full_graph_logratio','null_graph_logratio','relationship_excess','supported_context_fraction','confirming_blocks')+NAMES
FLOAT32_MAX=float(np.finfo(np.float32).max)

@njit(cache=True)
def channels(u,p1,p4,e2):
    u2=u*u
    return np.array([u,u2,1. if abs(u)>.9 else 0.,u*u2,u*p1,u*p4,(u2-e2)*(p1*p1-e2),(u2-e2)*(p4*p4-e2)])

@njit(cache=True)
def raw_replay(points,sorted_h,initial,e2):
    ring=initial.copy();out=np.empty((len(points),8));ranks=np.empty(len(points))
    for t in range(len(points)):
        u=2.*ecdf_uniform(points[t],sorted_h)-1.;out[t]=channels(u,ring[(t-1)%4],ring[t%4],e2);ring[t%4]=u;ranks[t]=u
    return out,ranks

def categorize(z,valid=True):
    z=np.asarray(z);s=np.ones(z.shape,dtype=np.int8);s[z<-2]=0;s[z>2]=2
    if not valid:s[:]=3
    s[~np.isfinite(z)]=3
    return s

@njit(cache=True)
def step(point,args):
    sorted_h,center,spread,ring,sums,info,memory,previous,node_z,params,parents,full,null,support,output=args
    t=info[0];u=2.*ecdf_uniform(point,sorted_h)-1.
    values=channels(u,ring[(t-1)%4],ring[t%4],params[2]);ring[t%4]=u;sums+=values
    raw_shock=abs(point-params[0])>8.*params[1]
    if info[4]>0:info[4]-=1
    if raw_shock:info[4]=16
    output[3]=1. if raw_shock else 0.;output[4]=0.
    complete=(t+1)%16==0
    if complete:
        b=info[1];info[1]+=1
        current=np.empty(8,dtype=np.int8)
        for j in range(8):
            node_z[j]=min(max((sums[j]/16.-center[j])/spread[j],-12.),12.);sums[j]=0.;memory[b%4,j]=node_z[j]
            current[j]=0 if node_z[j]<-2 else 2 if node_z[j]>2 else 1
            if params[4]<.5:current[j]=3
        severity=0.;dominant=0
        if b>=3:
            for j in range(8):
                sign=memory[0,j]>0.;v=abs(memory[0,j])
                for k in range(1,4):
                    if (memory[k,j]>0.)!=sign:v=0.
                    v=min(v,abs(memory[k,j]))
                if v>severity:severity=v;dominant=j
        current_severity=np.max(np.abs(node_z));output[1]=severity;output[2]=current_severity;output[6]=dominant
        output[4]=1.;output[5]=info[1];output[11]=4. if b>=3 and severity>params[3] else 0.
        lf=ln=0.;used=supported=0
        for j in range(8):
            i=parents[j]
            if i<0:continue
            own=int(previous[j]);p=int(previous[i]);s=int(current[j])
            if own>=3 or p>=3 or s>=3:continue
            lf+=np.log(full[1,j,own,p,s]/full[0,j,own,p,s]);ln+=np.log(null[1,j,own,p,s]/null[0,j,own,p,s]);used+=1
            supported+=support[0,j,own,p]>=8 and support[1,j,own,p]>=8
        output[7]=lf/max(used,1);output[8]=ln/max(used,1);output[9]=(lf-ln)/max(used,1);output[10]=supported/max(used,1)
        previous[:]=current
        if params[4]<.5:info[2]=6
        elif severity>params[3] and b>=3:info[2]=4;info[3]=0
        elif info[2]==4:info[2]=5;info[3]=2
        elif info[2]==5 and info[3]>1:info[3]-=1
        else:
            info[3]=0;info[2]=2 if info[4]>0 else 3 if current_severity>=2. else 1
    elif params[4]<.5:info[2]=6
    elif info[1]>0 and info[2]!=4 and info[2]!=5:
        if info[4]>0:info[2]=2
        elif info[2]==2:info[2]=3 if output[2]>=2. else 1
    output[0]=info[2];output[12:20]=node_z;info[0]+=1
    return output

@njit(cache=True)
def replay_kernel(points,args):
    out=np.empty((len(points),20))
    for t in range(len(points)):out[t]=step(float(points[t]),args)
    return out

class State:
    def __init__(self,historical,*,threshold=np.inf,graph=None):
        h=np.asarray(historical,dtype=np.float64)
        if h.ndim!=1 or len(h)<512 or not np.isfinite(h).all() or np.any(np.abs(h)>FLOAT32_MAX):raise ValueError('Finite float32 H with at least512 observations required')
        if np.isnan(threshold) or threshold<0:raise ValueError('Nonnegative fixed evidence threshold required')
        h=h.astype(np.float32);fit=int(.6*len(h));sorted_h=np.sort(h[:fit].astype(float));initial=np.array([2*ecdf_uniform(float(x),sorted_h)-1 for x in h[fit-4:fit]])
        e2=float(np.mean(np.array([2*ecdf_uniform(float(x),sorted_h)-1 for x in h[:fit]])**2))
        raw,ranks=raw_replay(h[fit:],sorted_h,initial,e2);n=len(raw)//16;blocks=raw[:n*16].reshape(n,16,8).mean(axis=1)
        center=np.median(blocks,axis=0);q25,q75=np.quantile(blocks,[.25,.75],axis=0);spread=np.maximum((q75-q25)/1.349,.02)
        valid=float(np.std(h[:fit])>1e-8);previous=categorize((raw[-16:].mean(axis=0)-center)/spread,bool(valid))
        med=float(np.median(h[:fit]));sd=float(np.std(h[:fit].astype(float)));mad=float(1.4826*np.median(np.abs(h[:fit].astype(float)-med)));scale=max(mad,.01*sd,1e-8)
        params=np.array([med,scale,e2,float(threshold),valid]);parents=np.full(8,-1,dtype=np.int64);full=np.full((2,8,3,3,3),1/3);null=full.copy();support=np.zeros((2,8,3,3),dtype=np.int32)
        if graph is not None:
            graph.validate();parents=graph.parents.copy();full=graph.full;null=graph.null;support=graph.support
        output=np.zeros(20);output[0]=0 if valid else 6
        self.args=(sorted_h,center,spread,ranks[-4:].copy(),np.zeros(8),np.array([0,0,0 if valid else 6,0,0],dtype=np.int64),np.zeros((4,8)),previous,np.zeros(8),params,parents,full,null,support,output)
        self.calibration={'H_fit_length':fit,'H_calibration_blocks':n,'reference_valid':bool(valid),'spread_floor_nodes':np.flatnonzero((q75-q25)/1.349<.02).tolist()}
        self.H_block_z=np.clip((blocks-center)/spread,-12,12);self.H_previous_state=previous.copy()
    def predict_one(self,point):
        value=float(point)
        if not np.isfinite(value) or abs(value)>FLOAT32_MAX:raise ValueError('Finite float32 current scalar required')
        return step(float(np.float32(value)),self.args).copy()
    def replay(self,points):
        p=np.asarray(points,dtype=np.float64)
        if p.ndim!=1 or not np.isfinite(p).all() or np.any(np.abs(p)>FLOAT32_MAX):raise ValueError('Finite float32 scalar stream required')
        return replay_kernel(p.astype(np.float32),self.args)
    @property
    def bounded_mutable_state_bytes(self):return sum(self.args[j].nbytes for j in (3,4,5,6,7,8,14))

def lineage():
    formulas=('u','u^2','I(abs(u)>.9)','u^3','u_t*u_(t-1)','u_t*u_(t-4)','(u_t^2-H_Eu2)*(u_(t-1)^2-H_Eu2)','(u_t^2-H_Eu2)*(u_(t-4)^2-H_Eu2)')
    return [{'node':j,'name':name,'korean':KOREAN[j],'family':FAMILIES[j],'formula':formulas[j],'aggregation':'mean of16 distinct current observations; no EWMA or CUSUM','raw_influence_horizon':'current through4 subsequent points at most','shared_raw':True,'H_reference':'first60% empirical CDF; last40% block calibration','lagged_macro_parent':'previous completed16-point block only','interpretation':'bounded rank feature; not independent external cause'} for j,name in enumerate(NAMES)]
