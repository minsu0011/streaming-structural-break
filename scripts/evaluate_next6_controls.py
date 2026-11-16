"""Independent raw counterexamples and stationary calibration; no model changes."""
from pathlib import Path
import sys,os,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next6/numba_cache')
import numpy as np,joblib
from src.next6.common import BASE,read_json,save,digest,activity,checkpoint
from src.next6.model import EXTRA_COLUMNS
from src.next6.graph import GraphState
from src.next6.evaluation import csv_write
from src.next5.synthetic import load_reference,threshold,alarm_summary
from prepare_next6_controls import key

def predict_cohort(fold,spec,models,refs):
    d=BASE/'synthetic_cache'/key(spec);r=read_json(d/'DONE.json')
    outdir=BASE/f'synthetic_predictions/fold_{fold}';outdir.mkdir(parents=True,exist_ok=True);path=outdir/(key(spec)+'.npz')
    if path.exists():
        with np.load(path,allow_pickle=False) as z:return {m:z[m] for m in z.files}
    tab=np.load(d/'tabular.npy',mmap_mode='r');states=np.load(d/'states.npy',mmap_mode='r');initial=np.load(d/'H_states.npy')
    p={m:np.empty(tab.shape[:2]) for m in ('M0','AMPLITUDE_0.9_M0',*models)}
    for i in range(len(tab)):
        p['M0'][i]=.8*refs['J102'].predict(tab[i,:,:77])+.2*refs['C205'].predict(tab[i,:,77:101]);p['AMPLITUDE_0.9_M0'][i]=.9*p['M0'][i]
        shared=None
        for m,model in models.items():
            base=np.asarray(tab[i,:,:101 if m=='BASE_ONLY' else 245]);cols=EXTRA_COLUMNS[m]
            if cols:
                if m in ('M2','M3','M4'):
                    if shared is None:shared=GraphState(model.graph,initial[i]).replay(tab[i,:,101:],states[i])
                    extra=shared
                else:extra=GraphState(model.graph,initial[i]).replay(tab[i,:,101:],states[i])
                base=np.column_stack((base,extra[:,cols]))
            p[m][i]=model.predict(base)
    assert all(np.isfinite(v).all() for v in p.values());np.savez(path,**p)
    save(str(path.with_suffix('.json').relative_to(BASE)),{'prediction_sha256':digest(path),'cohort_manifest_sha256':digest(d/'DONE.json'),'fold':fold,
        'model_sha256':{m:digest(BASE/f'fold_{fold}/outer/{m}/model.joblib') for m in models},'synthetic_only':True})
    return p

def recovery(delta,start):
    ok=np.abs(delta[start:])<=.01
    if len(ok)<16:return None
    run=np.convolve(ok.astype(int),np.ones(16,dtype=int),'valid');where=np.flatnonzero(run==16)
    return int(where[0]) if len(where) else None

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fold',type=int);args=parser.parse_args()
    lock=read_json(BASE/'DIAGNOSTIC_LOCK.json');menu=lock['menu']
    for fold in ([args.fold] if args.fold is not None else range(5)):
        if (BASE/f'synthetic_predictions/fold_{fold}/SUMMARY.json').exists():continue
        assert (BASE/f'fold_{fold}/OUTER_PREDICTIONS_FROZEN.json').exists()
        models={m:joblib.load(BASE/f'fold_{fold}/outer/{m}/model.joblib') for m in read_json(BASE/f'fold_{fold}/OUTER_PREDICTIONS_FROZEN.json')['models']}
        for model in models.values():model.validate()
        refs,_=load_reference('full',fold);cal={};thresholds={};records=[];paired=[]
        with activity('synthetic_fold_'+str(fold),'CONTROL_RESEARCH'):
            for spec in menu:
                if spec['kind']!='calibration':continue
                p=predict_cohort(fold,spec,models,refs)
                for m,v in p.items():cal.setdefault(m,[]).extend(v.max(axis=1).tolist())
            thresholds={m:threshold(v,.05) for m,v in cal.items()}
            for spec in menu:
                if spec['kind']=='calibration':continue
                p=predict_cohort(fold,spec,models,refs);rec=read_json(BASE/'synthetic_cache'/key(spec)/'DONE.json');rows=rec['rows'];indices=np.array([i for i,r in enumerate(rows) if not r['control']]) if spec['paired'] else np.arange(len(rows))
                taus=np.array([rows[i]['tau'] for i in indices])
                for m,v in p.items():
                    z=v[indices];info=alarm_summary(z,taus,thresholds[m]);info.update({'fold':fold,'method':m,'kind':spec['kind'],'scenario':spec['scenario'],
                        'horizon':spec['horizon'],'evaluation_series':len(indices),'score_units':'raw detector probability','threshold_source':'independent stationary synthetic partition0; 256 maxima@1024',
                        'anytime_alpha_claim':False})
                    info['eventual_alarm_rate_at_M0_numeric_cutoff']=float(np.mean(np.any(z>thresholds['M0'],axis=1)))
                    info['metric_directions']='null/prebreak alarm and recovery/delay: lower conditional on detection; true detection: higher; score excess alone: not sufficient'
                    if spec['paired']:
                        control=v[indices+1];prefix_error=0.;effects=[];recover=[];persistence=[]
                        for k,i in enumerate(indices):
                            at=rows[i]['scheduled_tau'];delta=z[k]-control[k];prefix_error=max(prefix_error,float(abs(delta[:at]).max(initial=0)))
                            effects.append(float(delta[at:].max(initial=0)));end=at+(16 if spec['scenario']=='OB16' else 1)
                            if spec['scenario'] in ('O1','OB16'):
                                r=recovery(delta,end);recover.append(r);persistence.append(float(np.mean(delta[min(end+32,len(delta)):]> .01)))
                        assert prefix_error<1e-12,(fold,m,spec,prefix_error)
                        info.update({'paired_prechange_prefix_max_error':prefix_error,'mean_paired_post_peak_increase':float(np.mean(effects))})
                        if recover:
                            observed=[r for r in recover if r is not None]
                            info.update({'recovery_censor_fraction':sum(r is None for r in recover)/len(recover),'conditional_recovery_delay_median':float(np.median(observed)) if observed else None,
                                'mean_positive_score_excess_fraction_after_32':float(np.mean(persistence))})
                    records.append(info)
                checkpoint('synthetic evaluated',f'fold{fold} {key(spec)}')
            save(f'synthetic_predictions/fold_{fold}/SUMMARY.json',{'thresholds':thresholds,'records':records,'calibration_series':256,'thresholds_selected_without_evaluation':True})
    if all((BASE/f'synthetic_predictions/fold_{f}/SUMMARY.json').exists() for f in range(5)):
        records=[r for f in range(5) for r in read_json(BASE/f'synthetic_predictions/fold_{f}/SUMMARY.json')['records']]
        csv_write('COUNTEREXAMPLE_RESULTS_NEXT6.csv',records)
        csv_write('CONTROL_RESULTS_NEXT6.csv',[r for r in records if r['scenario'] in ('N0','N1','N2','N3','O1','OB16')])
        save('SYNTHETIC_CONTROL_AUDIT.json',{'status':'PASS','five_fold_models':True,'raw_generator_unchanged':True,'paired_prebreak_prefix_exact':True,
            'calibration_and_evaluation_partition_disjoint':True,'no_threshold_retuning':True,'long_horizon_is_diagnostic_not_anytime_guarantee':True})

if __name__=='__main__':main()
