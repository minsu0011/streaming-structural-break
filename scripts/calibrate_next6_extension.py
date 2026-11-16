from pathlib import Path
import sys,os,math,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next6_extension/numba_cache')
import numpy as np,joblib
from src.next6_extension.common import *
from src.next6_extension.fresh_state import State
from src.next6_extension.synthetic import generated,CASES
from src.next5.synthetic import load_reference
from src.next5.runtime import SharedResearchBaseline,baseline_step
from src.next6.runtime import Prepared
from numba import njit

@njit(cache=False)
def baseline_replay(points,args):
    out=np.empty(len(points),dtype=np.float64)
    for t in range(len(points)):out[t]=baseline_step(float(points[t]),args)
    return out

def fixed_models(fold):
    refs,_=load_reference('full',fold);m0=SharedResearchBaseline(refs['J102'],refs['C205']);m3=Prepared(joblib.load(ROOT/f'artifacts/next6/fold_{fold}/outer/M3/model.joblib'),resident=False)
    return m0,m3

def baseline_predict(models,h,o):
    m0,m3=models;s=m0.new_state(h);p0=baseline_replay(o,s.args)
    return {'M0':p0.astype(float),'M3':m3.new_state(h).replay(o)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=range(5));p.add_argument('--fresh',action='store_true');a=p.parse_args();lock=read_json(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json')
    if not (BASE/'EXTENDED_SYNTHETIC_LOCK.json').exists():
        save('EXTENDED_SYNTHETIC_LOCK.json',{'created_utc':utc_now(),'generator_source_sha256':digest(ROOT/'src/next6_extension/synthetic.py'),'original_generator_lock_sha256':digest(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json'),
            'cases':list(CASES),'calibration_partition':30,'evaluation_partition':31,'calibration_per_background':256,'evaluation_per_case':512,'horizons':[1024,4096],
            'baseline_methods':['M0','M3'],'baseline_folds':list(range(5)),'baseline_reason':'Frozen champion and representative transition model only; no repeated benchmark of all rejected candidates.',
            'paired_extended_case_details':'OB16 alternating +/-8 for16; repeated4 +/-20 impulses every128; temporary x sqrt2 observation scale32; return-normal x sqrt2 for256; late V2 at1536; V2 with +/-20 shock at tau+32.',
            'temporary_labels':'Temporary scale changes keep breakOccurred=1 after first change, even when the observable state recovers. Not treated as no-break contamination.',
            'scalar_recovery':'16 consecutive equal fresh evidence points, or baseline scores within .01 of paired control; censoring reported. Strict all-future coupling bound checked separately.',
            'paired_control_reuse':'Extended cases can share underlying control noise; per-case CIs, not an independence claim across cases.'})
    assert digest(ROOT/'src/next6_extension/synthetic.py')==read_json(BASE/'EXTENDED_SYNTHETIC_LOCK.json')['generator_source_sha256']
    name='fresh' if a.fresh else f'fold_{a.fold}';assert a.fresh or a.fold is not None;dest=BASE/'calibration'/name;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'DONE.json').exists():return
    models=None if a.fresh else fixed_models(a.fold);maxima={};rows=[]
    with activity('independent_normal_calibration_'+name,'CALIBRATION'):
        for case in ('N0','N1','N2','N3'):
            for seed in range(256):
                h,o,tau,info=generated(case,seed,partition=30,horizon=1024,control=True,lock=lock)
                pp={'FRESH':State(h).replay(o)[:,1]} if a.fresh else baseline_predict(models,h,o)
                for method,values in pp.items():
                    maximum=float(np.max(values));maxima.setdefault(method,[]).append(maximum);rows.append({'background':case,'seed':seed,'method':method,'maximum':maximum})
            checkpoint('calibration completed background',name+' '+case)
        cut={m:float(np.sort(v)[math.ceil((len(v)+1)*.95)-1]) for m,v in maxima.items()}
        csv_write(f'calibration/{name}/MAXIMA.csv',rows);save(f'calibration/{name}/DONE.json',{'status':'FROZEN','created_utc':utc_now(),'thresholds':cut,'independent_null_series':1024,'horizon':1024,'partition':30,'rank':math.ceil(1025*.95),'strict_greater':True,'evaluation_partition_opened':False})
if __name__=='__main__':main()
