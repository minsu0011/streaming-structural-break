"""Transport diagnostics for the already frozen finite-horizon threshold."""
from pathlib import Path
import sys,os,copy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next6_extension/numba_cache')
import numpy as np
from src.next6_extension.common import *
from src.next6_extension.fresh_runtime import SafeState
from src.next2.counterexamples import generate

def main():
    path=BASE/'CALIBRATION_TRANSPORT_LOCK.json'
    if not path.exists():save(path.name,{'created_utc':utc_now(),'H_lengths':[1000,1500,3000,5000],'backgrounds':['N0','N1','N2','N3'],'seeds':512,'horizons':[1024,4096],
        'partition':33,'H_contamination_fractions':[0.,.01],'contamination':'Deterministic1% equally spaced H observations, alternating +/-20 stationary SD; sensitivity only, not assumed normal H.',
        'threshold':'Already frozen pooled H2048 partition30 threshold; never recalibrated on transport results.','purpose':'Quantify H-length/contamination transport and finite-horizon limits, not add or select a candidate.',
        'source_sha256':digest(Path(__file__)),'runtime_sha256':digest(ROOT/'src/next6_extension/fresh_runtime.py')})
    assert read_json(path)['source_sha256']==digest(Path(__file__))
    original=read_json(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json');cut=read_json(BASE/'calibration/fresh/DONE.json')['thresholds']['FRESH'];rows=[]
    with activity('frozen_threshold_H_transport','CALIBRATION_VALIDITY'):
        for nh in (1000,1500,3000,5000):
            local=copy.deepcopy(original);local['historical_length']=nh
            for case in ('N0','N1','N2','N3'):
                for seed in range(512):
                    h,o,tau=generate(case,seed,partition=33,horizon=4096,control=True,lock=local)
                    for fraction in (0.,.01):
                        hh=h.copy()
                        if fraction:
                            ix=np.linspace(0,len(hh)-1,max(1,int(len(hh)*fraction)),dtype=int);hh[ix]+=20*np.where(np.arange(len(ix))%2==0,1,-1)
                        out=SafeState(hh,threshold=cut).replay(o)
                        for horizon in (1024,4096):rows.append({'H_length':nh,'background':case,'seed':seed,'H_contamination_fraction':fraction,'horizon':horizon,'alarm':bool(np.any(out[:horizon,1]>cut)),'max_evidence':float(out[:horizon,1].max()),'threshold':cut})
                checkpoint('H transport cohort',str(nh)+' '+case)
        csv_write('CALIBRATION_TRANSPORT_SERIES.csv',rows);save('CALIBRATION_TRANSPORT_DONE.json',{'status':'COMPLETE','created_utc':utc_now(),'evaluated_series':len(rows)//2,'fixed_threshold':cut,'no_recalibration':True})
if __name__=='__main__':main()
