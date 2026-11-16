from pathlib import Path
import sys,os,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next6_extension/numba_cache')
import numpy as np
from src.next6_extension.common import *
from src.next6_extension.fresh_state import State,COLUMNS
from src.next6_extension.synthetic import generated,CASES,metrics
from calibrate_next6_extension import fixed_models,baseline_predict

def run(fold=None):
    name='fresh' if fold is None else f'fold_{fold}';cal=read_json(BASE/f'calibration/{name}/DONE.json');lock=read_json(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json');menu=read_json(BASE/'EXTENDED_SYNTHETIC_LOCK.json')
    assert digest(ROOT/'src/next6_extension/synthetic.py')==menu['generator_source_sha256']
    models=None if fold is None else fixed_models(fold);thresholds=cal['thresholds'];dest=BASE/'extended_synthetic'/name;dest.mkdir(parents=True,exist_ok=True)
    for case in CASES:
        receipt=dest/(case+'_DONE.json')
        if receipt.exists():continue
        rows=[];cases=[];state_stats=[]
        with activity('extended_counterexample_'+name+'_'+case,'PAIRED_MECHANISM_EVALUATION'):
            for seed in range(512):
                h,o,tau,info=generated(case,seed,partition=31,horizon=4096,control=False,lock=lock);hc,oc,_,_=generated(case,seed,partition=31,horizon=4096,control=True,lock=lock)
                assert np.array_equal(h,hc);assert np.array_equal(o[:min(info['scheduled'],4096)],oc[:min(info['scheduled'],4096)])
                if fold is None:
                    a=State(h,threshold=thresholds['FRESH']).replay(o);b=State(h,threshold=thresholds['FRESH']).replay(oc);p={'FRESH':a[:,1]};c={'FRESH':b[:,1]}
                    end=info['intervention_end']
                    if case in ('O1','OB16','REPEATED_SHOCK') and end+128<4096:assert np.array_equal(a[end+128:],b[end+128:]),(case,seed)
                    for horizon in (1024,4096):
                        for s in range(7):state_stats.append({'scenario':case,'seed':seed,'horizon':horizon,'state':s,'fraction':float(np.mean(a[:horizon,0]==s))})
                    if seed<3:cases.append({'seed':seed,'tau':tau,**info,'columns':COLUMNS,'treatment':a.tolist(),'control':b.tolist()})
                else:p=baseline_predict(models,h,o);c=baseline_predict(models,h,oc)
                for method,values in p.items():
                    for horizon in (1024,4096):
                        row={'scenario':case,'seed':seed,'fold':fold,'method':method,**metrics(values,c[method],threshold=thresholds[method],tau=tau,info=info,horizon=horizon,units='fresh block evidence' if fold is None else 'raw detector probability')}
                        assert row['paired_prechange_max_error']<1e-12,(case,seed,method,row['paired_prechange_max_error']);rows.append(row)
                if (seed+1)%64==0:checkpoint('paired mechanism cohort',name+' '+case+' '+str(seed+1)+'/512')
            csv_write(f'extended_synthetic/{name}/{case}_SERIES.csv',rows)
            if fold is None:
                csv_write(f'extended_synthetic/{name}/{case}_STATE_OCCUPANCY.csv',state_stats)
                # Only three preregistered seed examples; derived states, no raw.
                save(f'extended_synthetic/{name}/{case}_EXAMPLES.json',{'scenario':case,'case_selection':'first3 seed indices fixed before scoring','curves':cases})
            save(str(receipt.relative_to(BASE)),{'status':'COMPLETE','created_utc':utc_now(),'scenario':case,'seeds':512,'calibration_sha256':digest(BASE/f'calibration/{name}/DONE.json'),'paired_prechange_exact':True,'source_sha256':digest(Path(__file__)),'no_tuning':True})
    save(f'extended_synthetic/{name}/DONE.json',{'status':'COMPLETE','created_utc':utc_now(),'cases':list(CASES),'seeds_per_case':512,'paired':True,'horizons':[1024,4096]})

def main():
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=range(5));p.add_argument('--fresh',action='store_true');a=p.parse_args();assert a.fresh or a.fold is not None;run(None if a.fresh else a.fold)
if __name__=='__main__':main()
