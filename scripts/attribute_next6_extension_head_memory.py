"""Exact four-group paired replacements in frozen M3, explicitly not causal SHAP."""
from pathlib import Path
import sys,os,itertools,math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next6_extension/numba_cache')
import numpy as np,joblib
from src.next6_extension.common import *
from src.next6_extension.synthetic import generated
from src.next5.reference_features import ReferenceState
from src.next6.nodes import NodeState,SET_C
from src.next6.graph import GraphState

GROUPS=('J102_base','C205_base','node_temporal','graph_transition');SLICES=(slice(0,77),slice(77,101),slice(101,245),slice(245,257))

def base(h,o):
    j=ReferenceState(h,{'version':'next5_reference_features_v1','kind':'J102'});c=ReferenceState(h,SET_C);n=NodeState(h);initial=n.pseudo_s[-16:].copy();x,s,t=n.replay(o)
    return np.column_stack((j.replay(o),c.replay(o),t)),s,initial

def main():
    path=BASE/'HEAD_MEMORY_ATTRIBUTION_LOCK.json'
    if not path.exists():save(path.name,{'created_utc':utc_now(),'cases':['O1','OB16','V2','D2'],'seeds':64,'folds':list(range(5)),'horizon':4096,'partition':31,
        'groups':GROUPS,'method':'All16 coalitions, exact4-group Shapley averaging on probability differences between paired treatment/control feature rows.',
        'interpretation':'Computational group replacement sensitivity. Hybrid feature rows can violate deterministic feature relations and be off-manifold; no causal feature-attribution claim.',
        'windows_relative_to_intervention':[[0,7],[8,31],[32,127],[128,511],[512,2047],[2048,4095]],'models':'Frozen M3 only, no fits or feature selection.','source_sha256':digest(Path(__file__))})
    assert read_json(path)['source_sha256']==digest(Path(__file__))
    models=[joblib.load(ROOT/f'artifacts/next6/fold_{f}/outer/M3/model.joblib') for f in range(5)]
    for model in models:model.validate()
    lock=read_json(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json');windows=read_json(path)['windows_relative_to_intervention'];maxerror=0.
    for case in ('O1','OB16','V2','D2'):
        dest=BASE/'head_memory'/case;dest.mkdir(parents=True,exist_ok=True)
        if (dest/'DONE.json').exists():continue
        rows=[];examples=[]
        with activity('frozen_head_memory_'+case,'BOTTLENECK_ATTRIBUTION'):
            for seed in range(64):
                h,o,tau,info=generated(case,seed,partition=31,horizon=4096,control=False,lock=lock);_,oc,_,_=generated(case,seed,partition=31,horizon=4096,control=True,lock=lock);at=info['scheduled']
                xb,sb,ib=base(h,o);xc,sc,ic=base(h,oc)
                for fold,model in enumerate(models):
                    g=model.graph;b=np.column_stack((xb,GraphState(g,ib).replay(xb[:,101:],sb)[:,12:]));c=np.column_stack((xc,GraphState(g,ic).replay(xc[:,101:],sc)[:,12:]));coalitions=np.empty((16,len(o)))
                    for mask in range(16):
                        x=c.copy()
                        for group,sl in enumerate(SLICES):
                            if mask&(1<<group):x[:,sl]=b[:,sl]
                        coalitions[mask]=model.predict(x)
                    phi=np.zeros((4,len(o)))
                    for j in range(4):
                        for mask in range(16):
                            if mask&(1<<j):continue
                            size=mask.bit_count();weight=math.factorial(size)*math.factorial(3-size)/math.factorial(4);phi[j]+=weight*(coalitions[mask|(1<<j)]-coalitions[mask])
                    err=float(np.max(np.abs(phi.sum(axis=0)-(coalitions[15]-coalitions[0]))));maxerror=max(maxerror,err);assert err<1e-12
                    assert np.max(np.abs(phi[:,:at]),initial=0)<1e-12
                    for low,high in windows:
                        start=at+low;end=min(at+high+1,len(o))
                        if start>=end:continue
                        for j,name in enumerate(GROUPS):rows.append({'scenario':case,'seed':seed,'fold':fold,'relative_start':low,'relative_end':high,'observed_points':end-start,'group':name,'mean_signed_contribution':float(phi[j,start:end].mean()),'mean_absolute_contribution':float(np.abs(phi[j,start:end]).mean()),'actual_mean_prediction_effect':float((coalitions[15,start:end]-coalitions[0,start:end]).mean()),'off_manifold_hybrid_diagnostic':True})
                    if seed<3 and fold==0:examples.append({'seed':seed,'tau':tau,'scheduled':at,'groups':GROUPS,'contribution':phi.T.tolist(),'treatment':coalitions[15].tolist(),'control':coalitions[0].tolist()})
                if (seed+1)%8==0:checkpoint('paired group replacement',case+' '+str(seed+1)+'/64')
            csv_write(f'head_memory/{case}/SERIES_WINDOW_ATTRIBUTION.csv',rows);save(f'head_memory/{case}/EXAMPLES.json',{'cases':examples,'hybrid_rows_not_causal_interventions':True})
            save(f'head_memory/{case}/DONE.json',{'status':'COMPLETE','created_utc':utc_now(),'max_additivity_error':maxerror,'seeds':64,'folds':5,'no_refit':True})
    save('HEAD_MEMORY_ATTRIBUTION_DONE.json',{'status':'COMPLETE','created_utc':utc_now(),'max_additivity_error':maxerror})
if __name__=='__main__':main()
