"""Registered empirical uncertainty of already observed long-late alarm diagnostics."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next2.alarm_uncertainty import METRICS,trajectory_maxima,event_matrix,bootstrap_thresholds,bootstrap_events
from src.next2.synthetic_analysis import threshold,age_normalize
from src.next2.session import checkpoint
from src.utils.artifacts import write_json,sha256,utc_now

J='J102_SERIAL_VARIANCE_512';Y='Y001_LONG_VARIANCE_ARCH1';R='B104_LONG_VARIANCE_NO_GLR_MEMORY'
C='C205_DEPENDENCE';B='BL2_C205_DEPENDENCE_80'
MODELS=(J,Y,R,C,B)
OUT=ROOT/'artifacts/next2/alarm_uncertainty'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    late_path=ROOT/'configs/next2/LATE_ALARM_DIAGNOSTIC_LOCK.json'
    late=json.loads(late_path.read_text())['identity']
    identity={'late_lock_sha256':sha256(late_path),'replicates':2000,'calibration_seed':2026090953,
        'evaluation_seed':2026090957,'models':list(MODELS),'betas':[0.,.5,1.],'alphas':[.01,.05],
        'source_sha256':{p:sha256(ROOT/p) for p in ('src/next2/alarm_uncertainty.py','src/next2/synthetic_analysis.py',
            'docs/NEXT2_ALARM_UNCERTAINTY_PROTOCOL.md','scripts/bootstrap_next2_alarm_uncertainty.py')},
        'new_fits':0,'promotion_authority':False,'registered_after_point_estimates':True}
    lock=ROOT/'configs/next2/ALARM_UNCERTAINTY_LOCK.json'
    if lock.exists():
        assert json.loads(lock.read_text())['identity']==identity
    else:
        write_json(lock,{'created_utc':utc_now(),'identity':identity})
    all_rows=[];all_contrasts=[];sources={}
    for cell_ordinal,cell in enumerate(late['cells']):
        bg,onset=cell['background'],cell['onset'];bg_ordinal=('N0','N2').index(bg)
        label=bg+'_ONSET_'+str(onset);folder=OUT/label;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'RESULTS.json').exists():
            r=json.loads((folder/'RESULTS.json').read_text());assert r['identity']==identity
            for name,digest in r['files_sha256'].items():assert sha256(folder/name)==digest
            all_rows.append(pd.read_csv(folder/'INTERVALS.csv',float_precision='round_trip'))
            all_contrasts.append(pd.read_csv(folder/'PAIRED_CONTRASTS.csv',float_precision='round_trip'))
            sources[folder.relative_to(ROOT).as_posix()+'/RESULTS.json']=sha256(folder/'RESULTS.json')
            continue
        original=ROOT/'artifacts/next2/late_alarm_diagnostic'/label
        receipt=json.loads((original/'RESULTS.json').read_text())
        assert receipt['status']=='PASS' and receipt['identity']==late
        analysis=json.loads((original/'ANALYSIS.json').read_text())
        assert analysis['table_sha256']==sha256(original/'ANALYSIS.csv')
        assert analysis['raw_result_sha256']==sha256(original/'RESULTS.json')
        observed=pd.read_csv(original/'ANALYSIS.csv',float_precision='round_trip')
        calibration_indices=np.random.default_rng(2026090953+bg_ordinal).integers(0,1000,(2000,1000),dtype=np.int32)
        pair_indices=np.random.default_rng(2026090957+cell_ordinal).integers(0,128,(2000,128),dtype=np.int32)
        vectors={};estimates={};levels_cache={};maxima_cache={};input_hashes={}
        for eid in MODELS:
            path=original/(eid+'.npy');assert sha256(path)==receipt['files_sha256'][eid+'.npy']
            input_hashes[path.relative_to(ROOT).as_posix()]=sha256(path)
            p=np.load(path,mmap_mode='r',allow_pickle=False)
            assert p.shape==(256,65536)
            calibration=ROOT/('artifacts/next2/finalist_counterexamples' if eid in (C,B) else 'artifacts/next2/counterexamples')/'calibration'/bg
            cr=json.loads((calibration/'RESULTS.json').read_text());cp_path=calibration/(eid+'.npy')
            assert cr['status']=='PASS' and sha256(cp_path)==cr['files_sha256'][eid+'.npy']
            input_hashes[cp_path.relative_to(ROOT).as_posix()]=sha256(cp_path)
            cp=np.load(cp_path,mmap_mode='r',allow_pickle=False);assert cp.shape==(1000,1024)
            for beta in identity['betas']:
                cal_max=age_normalize(cp,np.arange(1024),beta).max(axis=1)
                maxima=trajectory_maxima(age_normalize(p,np.arange(65536),beta),onset)
                key=eid+'_B'+str(beta);maxima_cache[key]=maxima
                for alpha in identity['alphas']:
                    slot=(eid,beta,alpha);level=threshold(cal_max,alpha)
                    estimate=event_matrix(maxima,level).mean(axis=0)
                    expected=observed[(observed.candidate==eid)&(observed.beta==beta)&(observed.alpha==alpha)]
                    assert len(expected)==1 and level==expected.iloc[0].threshold
                    np.testing.assert_array_equal(estimate,expected.iloc[0][list(METRICS)].to_numpy(dtype=float))
                    levels=bootstrap_thresholds(cal_max,calibration_indices,alpha)
                    vectors[slot]=bootstrap_events(maxima,levels,pair_indices);estimates[slot]=estimate
                    levels_cache[key+'_A'+str(alpha)]=levels
            print('NEXT2 ALARM UNCERTAINTY',label,eid,flush=True)
        rows=[];contrasts=[]
        for slot,vector in vectors.items():
            eid,beta,alpha=slot
            base={'candidate':eid,'background':bg,'onset':onset,'beta':beta,'alpha':alpha,
                'replicates':2000,'calibration_series':1000,'evaluation_pairs':128,'promotion_authority':False}
            for k,metric in enumerate(METRICS):
                lo,hi=np.quantile(vector[:,k],[.025,.975])
                rows.append({**base,'metric':metric,'estimate':estimates[slot][k],'ci_2_5':float(lo),'ci_97_5':float(hi),
                    'bootstrap_sd':float(vector[:,k].std(ddof=1)),
                    'all_bootstrap_values_zero':bool(np.all(vector[:,k]==0)),
                    'uncertainty':'empirical calibration and matched evaluation-pair resampling; fixed fitted models'})
            refs=[(J,0.,alpha)]
            if beta!=0. and eid!=J:refs.append((eid,0.,alpha))
            for ref in refs:
                if ref==slot:continue
                difference=vector-vectors[ref];point=estimates[slot]-estimates[ref]
                for k,metric in enumerate(METRICS):
                    lo,hi=np.quantile(difference[:,k],[.025,.975])
                    contrasts.append({**base,'reference':ref[0],'reference_beta':ref[1],'metric':metric,
                        'estimate_delta':float(point[k]),'ci_2_5':float(lo),'ci_97_5':float(hi),
                        'bootstrap_fraction_delta_positive':float((difference[:,k]>0).mean())})
        rows=pd.DataFrame(rows);contrasts=pd.DataFrame(contrasts)
        rows.to_csv(folder/'INTERVALS.csv',index=False);contrasts.to_csv(folder/'PAIRED_CONTRASTS.csv',index=False)
        np.savez_compressed(folder/'BOOTSTRAP_VECTORS.npz',**{eid+'_B'+str(beta)+'_A'+str(alpha):value for (eid,beta,alpha),value in vectors.items()})
        np.savez_compressed(folder/'BOOTSTRAP_THRESHOLDS.npz',**levels_cache)
        np.savez_compressed(folder/'TRAJECTORY_MAXIMA.npz',**maxima_cache)
        files=('INTERVALS.csv','PAIRED_CONTRASTS.csv','BOOTSTRAP_VECTORS.npz','BOOTSTRAP_THRESHOLDS.npz','TRAJECTORY_MAXIMA.npz')
        write_json(folder/'RESULTS.json',{'status':'PASS','identity':identity,'created_utc':utc_now(),'cell':cell,
            'point_estimate_parity':True,'input_sha256':input_hashes,'new_fits':0,'seal_rows':0,
            'files_sha256':{f:sha256(folder/f) for f in files}})
        all_rows.append(rows);all_contrasts.append(contrasts)
        sources[folder.relative_to(ROOT).as_posix()+'/RESULTS.json']=sha256(folder/'RESULTS.json')
        checkpoint('Two-stage alarm bootstrap '+label,['Retain late-detection and calibration uncertainty','Complete isolated engineering measurements'])
    pd.concat(all_rows,ignore_index=True).to_csv(OUT/'INTERVALS.csv',index=False)
    pd.concat(all_contrasts,ignore_index=True).to_csv(OUT/'PAIRED_CONTRASTS.csv',index=False)
    write_json(OUT/'RESULTS.json',{'status':'PASS','identity':identity,'created_utc':utc_now(),'cells':len(all_rows),
        'receipts_sha256':sources,'tables_sha256':{f:sha256(OUT/f) for f in ('INTERVALS.csv','PAIRED_CONTRASTS.csv')},
        'limitation':'Empirical fixed-model diagnostic intervals; not post-selection coverage. Zero resampled detection does not imply a zero population probability.'})


if __name__=='__main__':main()
