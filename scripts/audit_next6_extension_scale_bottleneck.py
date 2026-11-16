"""Describe why a fixed four-block rank detector misses scale contractions.

No alternative detector, threshold, calibration or selected recipe is produced.
All synthetic cases are the already evaluated paired partition31 seeds.
"""
from pathlib import Path
import sys,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next6_extension/numba_cache')
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr,ndtri
from src.next6_extension.common import *
from src.next6_extension.fresh_runtime import SafeState
from src.next6_extension.fresh_state import NAMES
from src.next6_extension.synthetic import generated
from src.next6.data import source

def bounds(state):
    # Every possible empirical rank satisfies |u| <= n/(n+1).
    n=len(state.args[0]);a=n/(n+1);m=state.args[9][2];loenergy=-m;hienergy=a*a-m
    lo=np.array([-a,0,0,-a**3,-a*a,-a*a,min(loenergy*loenergy,loenergy*hienergy,hienergy*hienergy),min(loenergy*loenergy,loenergy*hienergy,hienergy*hienergy)])
    hi=np.array([a,a*a,1,a**3,a*a,a*a,max(loenergy*loenergy,loenergy*hienergy,hienergy*hienergy),max(loenergy*loenergy,loenergy*hienergy,hienergy*hienergy)])
    center,spread=state.args[1:3]
    return np.clip((lo-center)/spread,-12,12),np.clip((hi-center)/spread,-12,12)

def main():
    lock=BASE/'SCALE_BOTTLENECK_AUDIT_LOCK.json'
    if not lock.exists():save(lock.name,{'created_utc':utc_now(),'source_sha256':digest(Path(__file__)),'fixed_analysis':'H-only mathematical channel bounds on RP6400; same partition31 V1..V4 seeds0..511,1024/4096, single-block excursions and four-consecutive-same-sign excursions per node. No new alarms, thresholds, classifier recipes or exports.','interpretation':'A one-block excursion is a mechanism diagnostic, not a proposed one-block detector. All eight channels share raw observations. Ideal Gaussian integrals are explanatory population calculations, not empirical fitted H performance.'})
    assert read_json(lock)['source_sha256']==digest(Path(__file__))
    cut=read_json(BASE/'calibration/fresh/DONE.json')['thresholds']['FRESH'];data=source();low=[];high=[]
    with activity('scale_contraction_mechanism_audit','FIXED_REPRESENTATION_DIAGNOSTIC'):
        for sid in data.ids:
            h,_,_=data.get(int(sid));s=SafeState(h,threshold=cut);assert s.calibration['reference_valid'];a,b=bounds(s);low.append(a);high.append(b)
        low=np.array(low);high=np.array(high);hr=[]
        for j,name in enumerate(NAMES):
            hr.append({'node':j,'name':name,'H_series':6400,'negative_confirmation_mathematically_impossible_fraction':float(np.mean(low[:,j]>=-cut)),'positive_confirmation_mathematically_impossible_fraction':float(np.mean(high[:,j]<=cut)),'negative_bound_median':float(np.median(low[:,j])),'negative_bound_q05':float(np.quantile(low[:,j],.05)),'negative_bound_q95':float(np.quantile(low[:,j],.95)),'bound_is_conservative_not_joint_attainability':True})
        csv_write('RANK_CHANNEL_CONFIRMATION_LIMITS.csv',hr)
        rawlock=read_json(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json');series=[]
        for case in ('V1','V2','V3','V4'):
            for seed in range(512):
                h,o,tau,_=generated(case,seed,partition=31,horizon=4096,control=False,lock=rawlock);s=SafeState(h,threshold=cut);out=s.replay(o);end=np.flatnonzero(out[:,4]);z=out[end,12:20];a,b=bounds(s)
                assert np.all(z>=a-1e-12) and np.all(z<=b+1e-12)
                for horizon in (1024,4096):
                    post=(end>=tau+19)&(end<horizon);zz=z[post]
                    for j in range(8):
                        positive=zz[:,j]>cut;negative=zz[:,j]<-cut
                        p4=np.convolve(positive.astype(np.int8),np.ones(4,dtype=np.int8),'valid')==4 if len(zz)>=4 else np.array([],dtype=bool)
                        n4=np.convolve(negative.astype(np.int8),np.ones(4,dtype=np.int8),'valid')==4 if len(zz)>=4 else np.array([],dtype=bool)
                        series.append({'scenario':case,'seed':seed,'horizon':horizon,'node':j,'post_blocks':len(zz),'negative_single_block':bool(negative.any()),'positive_single_block':bool(positive.any()),'negative_four_new_blocks':bool(n4.any()),'positive_four_new_blocks':bool(p4.any()),'negative_block_fraction':float(negative.mean()),'positive_block_fraction':float(positive.mean()),'negative_impossible_from_H_bound':bool(a[j]>=-cut),'mean_post_z':float(zz[:,j].mean())})
            checkpoint('scale_contraction_mechanism_audit',case+' 512 reused seeds complete')
        csv_write('scale_bottleneck/SAME_EVALUATED_SEED_NODE_DIAGNOSTICS.csv',series)
        summary=[]
        for case in ('V1','V2','V3','V4'):
            for horizon in (1024,4096):
                for j,name in enumerate(NAMES):
                    rr=[r for r in series if r['scenario']==case and r['horizon']==horizon and r['node']==j];assert len(rr)==512
                    summary.append({'scenario':case,'horizon':horizon,'node':j,'name':name,'reused_seeds':512,**{key:float(np.mean([r[key] for r in rr])) for key in ('negative_single_block','positive_single_block','negative_four_new_blocks','positive_four_new_blocks','negative_block_fraction','positive_block_fraction','negative_impossible_from_H_bound','mean_post_z')},'conditional_post_window_not_official_first_detection_rate':True})
        csv_write('SCALE_DIRECTION_CONFIRMATION_DIAGNOSTIC.csv',summary)
        theory=[]
        phi=lambda x:np.exp(-x*x/2)/np.sqrt(2*np.pi)
        q=ndtri(.95)
        for variance in (.5,.7,1.,1.5,2.):
            s=np.sqrt(variance)
            energy,err=quad(lambda x:(2*ndtr(s*x)-1)**2*phi(x),-10,10,epsabs=1e-12,epsrel=1e-12)
            theory.append({'variance_ratio':variance,'ideal_rank_energy_mean':energy,'energy_change_from_null':energy-1/3,'energy_change_over_iid_null_16block_SD':(energy-1/3)/np.sqrt((1/5-1/9)/16),'ideal_rank_tail_probability':2*ndtr(-q/s),'tail_change_over_iid_null_16block_SD':(2*ndtr(-q/s)-.1)/np.sqrt(.1*.9/16),'integration_absolute_error_estimate':err,'reference':'Infinite uncontaminated N(0,1) CDF; iid online N(0,variance_ratio); finite-H IQR spread differs.'})
        csv_write('IDEAL_GAUSSIAN_SCALE_CHANNEL_EXPLANATION.csv',theory)
    data.guard.check_state();save('SCALE_BOTTLENECK_AUDIT_DONE.json',{'status':'COMPLETE','created_utc':utc_now(),'source_sha256':digest(Path(__file__)),'H_series':6400,'existing_evaluation_streams_replayed_for_diagnostics':2048,'no_new_detector_or_threshold':True,'bounded_channel_range_verified_on_replayed_points':True,'inference':'Channel floor/ceiling constraints and four-new-block requirement explain mechanism restrictions; they do not identify every source of empirical misses.'})

if __name__=='__main__':main()
