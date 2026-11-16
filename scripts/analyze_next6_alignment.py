"""Training-only pseudo-events and same-time not-yet-change matched contrasts."""
from pathlib import Path
import sys,os,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next6/numba_cache')
import numpy as np
from scipy.spatial import cKDTree
from src.next6.common import BASE,read_json,save,activity,checkpoint,ids_hash
from src.next6.data import arrays
from src.next6.evaluation import csv_write
from analyze_next6_atlas import correlation
from src.next6.nodes import NAMES,KOREAN

def main():
    with activity('pseudo_event_and_not_yet_matched_atlas','RELATION_ANALYSIS'):
        data,meta,v=arrays();train=np.flatnonzero(data.series['fold']!=0);tau=data.series['tau'];offset=data.series['online_offsets'];desc=np.asarray(v['H_descriptors'][:,:4]);mean=desc[train].mean(axis=0);sd=np.maximum(desc[train].std(axis=0),1e-6);z=(desc-mean)/sd
        null=[i for i in train if tau[i]<0];future=[i for i in train if tau[i]>=0];tree=cKDTree(z[future]);stats={};rows=[]
        phases=read_json(BASE/'NEXT6_PROTOCOL_LOCK.json')['phase_intervals']
        for ix in null:
            sid=int(data.ids[ix]);pseudo=32+int(hashlib.sha256(('NEXT6_PSEUDO:'+str(data.guard.split.groups[sid])).encode()).hexdigest(),16)%224
            a,b=offset[ix:ix+2];time=np.arange(b-a);age=time-pseudo
            for phase,(low,high) in phases.items():
                use=time[(age>=low)&(True if high is None else age<=high)]
                if len(use)>=4:
                    x=np.asarray(v['nodes'][a+use]);stats.setdefault('pseudo_null_'+phase,[]).append(correlation(x,x,True))
                rows.append({'kind':'pseudo_null','dataset_id':sid,'phase':phase,'pseudo_tau':pseudo,'available_rows':len(use),'control_id':sid})
        for n,ix in enumerate(future):
            sid=int(data.ids[ix]);a,b=offset[ix:ix+2];time=np.arange(b-a);age=time-tau[ix];dist,near=tree.query(z[ix],k=min(128,len(future)))
            for phase,(low,high) in phases.items():
                use=time[(age>=low)&(True if high is None else age<=high)]
                if len(use)<4:continue
                control=None
                for at in np.atleast_1d(near):
                    candidate=future[int(at)]
                    if candidate!=ix and tau[candidate]>use[-1] and offset[candidate+1]-offset[candidate]>use[-1]:control=candidate;break
                if control is None:
                    rows.append({'kind':'not_yet_matched','dataset_id':sid,'phase':phase,'available_rows':len(use),'matched':False});continue
                x=np.asarray(v['nodes'][a+use]);c=np.asarray(v['nodes'][offset[control]+use]);delta=correlation(x,x,True)-correlation(c,c,True)
                stats.setdefault('matched_not_yet_delta_'+phase,[]).append(delta)
                rows.append({'kind':'not_yet_matched','dataset_id':sid,'phase':phase,'available_rows':len(use),'matched':True,'control_id':int(data.ids[control]),'control_tau':int(tau[control]),'last_comparison_time':int(use[-1])})
            if n%500==0:checkpoint('not-yet matched atlas',f'{n}/{len(future)} train event series')
        out=[];rng=np.random.default_rng(20260913)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            for name,arr in stats.items():
                arr=np.array(arr);avg=np.nanmean(arr,axis=0);boot=np.array([np.nanmean(arr[rng.integers(len(arr),size=len(arr))],axis=0) for _ in range(200)]);lo,hi=np.nanquantile(boot,[.025,.975],axis=0)
                out.append({'name':name,'series':len(arr),'support':np.isfinite(arr).sum(axis=0).tolist(),'mean':np.nan_to_num(avg).tolist(),'CI_low':np.nan_to_num(lo).tolist(),'CI_high':np.nan_to_num(hi).tolist()})
        csv_write('CORRELATION_ATLAS/PSEUDO_AND_NOT_YET_SUPPORT.csv',rows)
        save('CORRELATION_ATLAS/PSEUDO_AND_NOT_YET.json',{'scope':'outer0 training only','training_ids_sha256':ids_hash(data.ids[train]),'results':out,
            'feature_names':list(NAMES),'korean_meaning':list(KOREAN),'same_absolute_online_time':True,'H_matching_fit_training_only':True,
            'pseudo_event_independent_of_labels_and_final_length':True,'not_yet_controls':'nearest admissible among fixed 128 H-neighbors; no future controls for late windows if unsupported',
            'uncertainty':'200 whole-event-series descriptive bootstrap; reused controls induce dependence; not confirmatory significance',
            'interpretation':'pre-event descriptive contrasts are not evidence of causal leading signals; abrupt synthetic prefix trap separately required'})

if __name__=='__main__':main()
