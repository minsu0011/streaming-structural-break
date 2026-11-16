"""Follow the measured J102 shock-memory bottleneck to fixed source primitives."""
from pathlib import Path
import sys,os,math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next6_extension/numba_cache')
import numpy as np,joblib
from src.next6_extension.common import *
from src.next6_extension.synthetic import generated
from src.next5.reference_features import feature_names
from src.next6.graph import GraphState
from attribute_next6_extension_head_memory import base
from summarize_next6_extension import rows

GROUPS={'ARCH_original_51':list(range(51)),'serial_current_and_recent':[i for i in range(51,77) if (i-51)%13<11],
        'serial_lifetime_max':[62,75],'serial_decayed_max':[63,76]}

def main():
    names=feature_names({'version':'next5_reference_features_v1','kind':'J102'});assert len(names)==77;assert all(names[i].endswith('memory_max') for i in GROUPS['serial_lifetime_max']);assert sorted(sum(GROUPS.values(),[]))==list(range(77))
    lock=BASE/'LEGACY_MEMORY_DETAIL_LOCK.json'
    if not lock.exists():save(lock.name,{'created_utc':utc_now(),'cases':['O1','OB16'],'same_previous_seeds':64,'folds':list(range(5)),'groups':{k:[{'column':i,'name':names[i]} for i in v] for k,v in GROUPS.items()},
        'method':'Exact four-subgroup Shapley replacement inside J102; all non-J102 inputs fixed at observed treatment. This is conditional computational sensitivity, not hierarchical SHAP or causal attribution.',
        'followup_reason':'Prior fixed four-group analysis localized late shock effect almost entirely to J102; trace its unbounded memory without changing any model.',
        'no_candidate_or_fit':True,'source_sha256':digest(Path(__file__))})
    assert read_json(lock)['source_sha256']==digest(Path(__file__))
    models=[joblib.load(ROOT/f'artifacts/next6/fold_{f}/outer/M3/model.joblib') for f in range(5)];original=read_json(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json');windows=read_json(BASE/'HEAD_MEMORY_ATTRIBUTION_LOCK.json')['windows_relative_to_intervention'];maxerror=0
    for case in ('O1','OB16'):
        dest=BASE/'legacy_memory'/case;dest.mkdir(parents=True,exist_ok=True)
        if (dest/'DONE.json').exists():continue
        output=[];feature=[];examples=[]
        with activity('legacy_memory_primitive_'+case,'BOTTLENECK_ATTRIBUTION'):
            for seed in range(64):
                h,o,tau,info=generated(case,seed,partition=31,horizon=4096,control=False,lock=original);_,oc,_,_=generated(case,seed,partition=31,horizon=4096,control=True,lock=original);at=info['scheduled'];xb,sb,ib=base(h,o);xc,sc,ic=base(h,oc)
                for low,high in windows:
                    a,b=at+low,min(at+high+1,len(o))
                    if a>=b:continue
                    for j,name in enumerate(names):feature.append({'scenario':case,'seed':seed,'relative_start':low,'relative_end':high,'column':j,'name':name,'mean_absolute_feature_difference':float(np.abs(xb[a:b,j]-xc[a:b,j]).mean()),'max_absolute_feature_difference':float(np.max(np.abs(xb[a:b,j]-xc[a:b,j])))})
                for fold,model in enumerate(models):
                    actual=np.column_stack((xb,GraphState(model.graph,ib).replay(xb[:,101:],sb)[:,12:]));values=np.empty((16,len(o)))
                    for mask in range(16):
                        hybrid=actual.copy();hybrid[:,:77]=xc[:,:77]
                        for k,columns in enumerate(GROUPS.values()):
                            if mask&(1<<k):hybrid[:,columns]=xb[:,columns]
                        values[mask]=model.predict(hybrid)
                    phi=np.zeros((4,len(o)))
                    for j in range(4):
                        for mask in range(16):
                            if mask&(1<<j):continue
                            k=mask.bit_count();phi[j]+=math.factorial(k)*math.factorial(3-k)/24*(values[mask|(1<<j)]-values[mask])
                    error=float(np.max(np.abs(phi.sum(axis=0)-(values[15]-values[0]))));assert error<1e-12;maxerror=max(maxerror,error)
                    for low,high in windows:
                        a,b=at+low,min(at+high+1,len(o))
                        if a>=b:continue
                        for j,group in enumerate(GROUPS):output.append({'scenario':case,'seed':seed,'fold':fold,'relative_start':low,'relative_end':high,'group':group,'mean_signed_contribution':float(phi[j,a:b].mean()),'mean_absolute_contribution':float(np.abs(phi[j,a:b]).mean()),'conditional_J102_effect':float((values[15,a:b]-values[0,a:b]).mean())})
                    if seed<3 and fold==0:examples.append({'seed':seed,'scheduled':at,'groups':list(GROUPS),'contribution':phi.T.tolist(),'conditional_effect':(values[15]-values[0]).tolist()})
                if (seed+1)%8==0:checkpoint('legacy source memory attribution',case+' '+str(seed+1)+'/64')
            csv_write(f'legacy_memory/{case}/SERIES_ATTRIBUTION.csv',output);csv_write(f'legacy_memory/{case}/FEATURE_DIFFERENCES.csv',feature);save(f'legacy_memory/{case}/EXAMPLES.json',{'cases':examples});save(f'legacy_memory/{case}/DONE.json',{'created_utc':utc_now(),'status':'COMPLETE','max_additivity_error':maxerror})
    save('LEGACY_MEMORY_DETAIL_DONE.json',{'status':'COMPLETE','created_utc':utc_now(),'max_additivity_error':maxerror,'interpretation':'Conditional off-manifold subgroup replacement; no model edits.'})
if __name__=='__main__':main()
