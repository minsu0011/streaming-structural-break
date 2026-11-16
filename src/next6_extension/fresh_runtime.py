"""Safe H initialization for the frozen fresh-state kernel.

Only the input adapter differs: H median/std use float64 accumulation. The frozen
scientific source and fitted graphs remain untouched. Ordinary-RP replay parity
is audited; the repair prevents overflow for finite extreme constant float32 H.
"""
import numpy as np
from src.next6_extension.fresh_state import State,raw_replay,categorize,FLOAT32_MAX
from src.next.normalization import ecdf_uniform

class SafeState(State):
    def __init__(self,historical,*,threshold=np.inf,graph=None):
        h=np.asarray(historical,dtype=np.float64)
        if h.ndim!=1 or len(h)<512 or not np.isfinite(h).all() or np.any(np.abs(h)>FLOAT32_MAX):raise ValueError('Finite float32 H with at least512 observations required')
        if np.isnan(threshold) or threshold<0:raise ValueError('Nonnegative fixed evidence threshold required')
        # Canonical float32 observations, then safe reduction dtype. Rank inputs
        # and every block/state equation remain exactly those of the frozen model.
        h=h.astype(np.float32);fit=int(.6*len(h));reference=h[:fit].astype(np.float64);sorted_h=np.sort(reference);initial=np.array([2*ecdf_uniform(float(x),sorted_h)-1 for x in h[fit-4:fit]])
        e2=float(np.mean(np.array([2*ecdf_uniform(float(x),sorted_h)-1 for x in h[:fit]])**2))
        raw,ranks=raw_replay(h[fit:],sorted_h,initial,e2);n=len(raw)//16;blocks=raw[:n*16].reshape(n,16,8).mean(axis=1)
        center=np.median(blocks,axis=0);q25,q75=np.quantile(blocks,[.25,.75],axis=0);spread=np.maximum((q75-q25)/1.349,.02)
        sd=float(np.std(reference));valid=float(sd>1e-8);previous=categorize((raw[-16:].mean(axis=0)-center)/spread,bool(valid))
        med=float(np.float32(np.median(reference)));mad=float(1.4826*np.median(np.abs(reference-med)));scale=max(mad,.01*sd,1e-8)
        params=np.array([med,scale,e2,float(threshold),valid]);parents=np.full(8,-1,dtype=np.int64);full=np.full((2,8,3,3,3),1/3);null=full.copy();support=np.zeros((2,8,3,3),dtype=np.int32)
        if graph is not None:
            graph.validate();parents=graph.parents.copy();full=graph.full;null=graph.null;support=graph.support
        output=np.zeros(20);output[0]=0 if valid else 6
        self.args=(sorted_h,center,spread,ranks[-4:].copy(),np.zeros(8),np.array([0,0,0 if valid else 6,0,0],dtype=np.int64),np.zeros((4,8)),previous,np.zeros(8),params,parents,full,null,support,output)
        self.calibration={'H_fit_length':fit,'H_calibration_blocks':n,'reference_valid':bool(valid),'spread_floor_nodes':np.flatnonzero((q75-q25)/1.349<.02).tolist()}
        self.H_block_z=np.clip((blocks-center)/spread,-12,12);self.H_previous_state=previous.copy()
