"""Sixteen exact existing feature coordinates, calibrated on pseudo-online H."""
import numpy as np
from numba import njit
from src.next5.reference_features import ReferenceState,feature_names

B_POS=np.array([0,2,11,15,51,52,53,64,77,81,85,89],dtype=np.int64)
C_POS=np.array([0,6,12,18],dtype=np.int64)
LAGS=(1,2,4,8,16)
SET_B={'version':'next5_reference_features_v1','kind':'B2_augmented_J'}
SET_C={'version':'next5_reference_features_v1','kind':'C205'}
NAMES=tuple(feature_names(SET_B)[j] for j in B_POS)+tuple(feature_names(SET_C)[j] for j in C_POS)
KOREAN=('부호 있는 혁신','강건 혁신 이탈','꼬리 초과 단기 증거','1시차 의존 단기 증거',
    '원 혁신 에너지','에너지 단기 지속','에너지 장기 지속','조건부 분산 예측 이탈',
    '원 에너지 하방','원 에너지 상방','조건부 에너지 하방','조건부 에너지 상방',
    '혁신 1시차 의존','혁신 2시차 의존','혁신 4시차 의존','혁신 8시차 의존')
FAMILIES=('innovation','innovation','tail','dependence','energy','energy','energy','variance',
    'energy','energy','conditional_energy','conditional_energy','dependence','dependence','dependence','dependence')

def select(b,c):return np.concatenate((b[...,B_POS],c[...,C_POS]),axis=-1)

@njit(cache=False)
def temporal_replay(x,history,ewma,counter):
    # Current, five true lags, first difference and two bounded EWMAs. H supplies warmup.
    out=np.empty((len(x),144),dtype=np.float32)
    for t in range(len(x)):
        p=counter[0]%16
        out[t,:16]=x[t]
        for k,lag in enumerate((1,2,4,8,16)):
            out[t,16+k*16:32+k*16]=history[(counter[0]-lag)%16]
        out[t,96:112]=x[t]-history[(counter[0]-1)%16]
        for k,half in enumerate((8.,32.)):
            alpha=1.-np.exp(-np.log(2.)/half)
            ewma[k]=(1.-alpha)*ewma[k]+alpha*x[t]
            out[t,112+k*16:128+k*16]=ewma[k]
        history[p]=x[t];counter[0]+=1
    return out

class NodeState:
    """Only H and successive scalar observations are accepted by this API."""
    def __init__(self,historical):
        h=np.asarray(historical,dtype=np.float32)
        if h.ndim!=1 or len(h)<96 or not np.isfinite(h).all():raise ValueError('Finite H >=96 required')
        self.fit_length=int(.6*len(h))
        self.b=ReferenceState(h[:self.fit_length],SET_B)
        self.c=ReferenceState(h[:self.fit_length],SET_C)
        pseudo=select(self.b.replay(h[self.fit_length:]),self.c.replay(h[self.fit_length:]))
        if not np.isfinite(pseudo).all():raise ValueError('Nonfinite pseudo-online H')
        burn=min(32,len(pseudo)//4);cal=pseudo[burn:]
        self.center=np.median(cal,axis=0).astype(np.float64)
        q25,q75=np.quantile(cal,[.25,.75],axis=0)
        self.spread=np.maximum((q75-q25)/1.349,.05).astype(np.float64)
        self.threshold=np.quantile(cal,[.2,.8],axis=0).astype(np.float64)
        self.degenerate=(self.threshold[1]-self.threshold[0]<1e-7)
        self.pseudo_raw=pseudo
        self.pseudo_x=self.normalize(pseudo)
        self.pseudo_s=self.categorize(pseudo)
        self.history=np.zeros((16,16),dtype=np.float32);self.ewma=np.zeros((2,16),dtype=np.float64)
        self.counter=np.zeros(1,dtype=np.int64)
        temporal_replay(self.pseudo_x,self.history,self.ewma,self.counter)

    def normalize(self,raw):return np.clip((raw-self.center)/self.spread,-12,12).astype(np.float32)

    def categorize(self,raw):
        # 0=low,1=normal,2=high,3=insufficient/missing. Ties to normal; never conflate missing with low.
        s=np.ones(np.asarray(raw).shape,dtype=np.int8)
        s[raw<self.threshold[0]]=0;s[raw>self.threshold[1]]=2
        s[...,self.degenerate]=3;s[~np.isfinite(raw)]=3
        return s

    def replay(self,points):
        p=np.asarray(points,dtype=np.float32)
        if p.ndim!=1 or not np.isfinite(p).all():raise ValueError('Finite scalar stream required')
        raw=select(self.b.replay(p),self.c.replay(p));x=self.normalize(raw);s=self.categorize(raw)
        temporal=temporal_replay(x,self.history,self.ewma,self.counter)
        return x,s,temporal

    def update(self,point):
        if not np.isscalar(point) or not np.isfinite(point):raise ValueError('Finite current scalar observation required')
        raw=select(self.b.update(float(point)).copy(),self.c.update(float(point)).copy())
        x=self.normalize(raw);s=self.categorize(raw)
        temporal=temporal_replay(x[None,:],self.history,self.ewma,self.counter)[0]
        return x,s,temporal

def lineage():
    formulas=[
        'ARCHCalibratedState output -> H baseline z; clipped signed innovation',
        'Same ARCH input -> (z-H_median)/(1.4826 H_MAD), clipped',
        '(EWMA_5[I(abs(z)>2)] - H_tail_rate)*sqrt(effective_n)/sqrt(max(p*(1-p),floor))',
        '(EWMA_5[z_t*z_t-1]-H_lag1_product)*sqrt(effective_n)',
        'H-calibrated ((residual-H_center)^2 / H_stationary_variance - 1)',
        'Clipped calibrated raw energy -> EWMA half-life8 / finite-t white-noise normalizer',
        'Clipped calibrated raw energy -> EWMA half-life32 / finite-t white-noise normalizer',
        'H-calibrated log((omega+alpha*q_t-1)/H_stationary_variance)',
        'max(median_H(log(energy+1e-4))-log(energy+1e-4),0)/H_lower_median, clip[0,12]',
        'max(log(energy+1e-4)-median_H(log(energy+1e-4)),0)/H_upper_median, clip[0,12]',
        'Same lower branch with squared residual / conditional variance',
        'Same upper branch with squared residual / conditional variance']
    formulas += [f'H-calibrated (conditional_z_t-H_mean)*(conditional_z_t-{lag}-H_mean) -> EWMA8 / finite-t normalizer' for lag in (1,2,4,8)]
    result=[]
    for i,(name,ko,family,formula) in enumerate(zip(NAMES,KOREAN,FAMILIES,formulas)):
        source='src/features/streaming.py' if i<4 else 'src/next/serial_variance_evidence.py' if i<8 else 'src/next4/scale_asymmetry.py' if i<12 else 'src/next2/dependence_runtime.py;src/next2/complement_bank.py'
        result.append({'node':i,'feature_name':name,'korean_meaning':ko,'raw_channel':'univariate observation -> H-fitted AR8/ARCH1 innovation',
            'formula':formula,'family':family,'source':source,'lag_window':{2:'EWMA5',3:'lag1 EWMA5',5:'half-life8',6:'half-life32',7:'lag1 ARCH forecast',12:'lag1 EWMA8',13:'lag2 EWMA8',14:'lag4 EWMA8',15:'lag8 EWMA8'}.get(i,'current'),
            'accumulation':'EWMA' if i in (2,3,5,6,12,13,14,15) else 'none (forecast uses prior residual)',
            'H_only_parameters':'AR/ARCH and engine references fit on first60% H; last40% H pseudo-online calibrates node median/IQR/q20/q80',
            'direct_transform':'MECHANICAL_DERIVED_OR_SHARED_INPUT','expected_mechanical_delay':'filter half-life/lag dependent; no economic causal interpretation',
            'warmup_missing':'H pseudo-online warmup, missing/degenerate state=3; nonfinite observations rejected',
            'normalization':'clip((raw_node-pseudo_H_median)/max(pseudo_H_IQR/1.349,.05),-12,12)'})
    return result
