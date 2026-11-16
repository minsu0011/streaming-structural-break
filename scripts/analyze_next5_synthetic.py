"""Fixed-model paired-seed stress analysis; calibration stays independent."""
from pathlib import Path
import argparse
import csv
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from numba import set_num_threads
from src.next5.artifacts import BASE,atomic_json,read_json,digest,utc_now,progress
from src.next5.guard import ResearchGuard
from src.next5.synthetic import cohort_menu,cohort_key,background_mapping,scenario_background,threshold,alarm_summary,load_candidate
from src.next5.synthetic_metrics import load_cohort,score,score_draws,reference_score_draws,event_draws
from src.next5.bootstrap import paired_summary
from src.next5.impulse_reports import impulse_tables


def write_table(path,rows):
    if not rows:return
    with path.open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',required=True);parser.add_argument('--scope',choices=('full','screen'),required=True)
    parser.add_argument('--fold',type=int,required=True);args=parser.parse_args();cid=args.candidate
    menu=read_json(BASE/'synthetic/MENU_LOCK.json');execution=read_json(BASE/'synthetic/EXECUTION_LOCK.json');execution_sha=digest(BASE/'synthetic/EXECUTION_LOCK.json')
    for name,sha in execution['sources_sha256'].items():assert digest(ROOT/name)==sha
    root=BASE/'synthetic/candidates'/cid/args.scope/('fold_'+str(args.fold));reference_root=BASE/'synthetic/references'/args.scope/('fold_'+str(args.fold))
    assert read_json(root/'REPLAY_RESULTS.json')['status']=='COMPLETE';out=root/'ANALYSIS';out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'START.json',{'created_utc':utc_now(),'candidate':cid,'scope':args.scope,'fold':args.fold,'execution_sha256':execution_sha,'new_fits':0},immutable=True)
    set_num_threads(2);guard=ResearchGuard();model,registration,model_proof=load_candidate(cid,args.scope,args.fold)
    lock=read_json(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json');cohorts=cohort_menu(menu,lock);backgrounds=background_mapping(lock)
    seed_lock=read_json(BASE/'synthetic/SEED_COUNTS_LOCK.json');seed_record=seed_lock['sizes'][str(menu['fold_model_sizes']['evaluation'])]
    count_path=ROOT/seed_record['path'];assert digest(count_path)==seed_record['sha256']
    with np.load(count_path,allow_pickle=False) as source:counts=source['counts'];seed_ids=source['synthetic_seed_index']
    assert np.array_equal(seed_ids,np.arange(counts.shape[1]))
    cutoffs={};calibration_rows=[];point_scores=[];paired_scores=[];counter_alarms=[];long_alarms=[];alarm_comparisons=[]
    impulse_lags=[];impulse_summary=[];impulse_comparisons=[];impulse_features=[];files={};cohort_sources=[]
    for spec in cohorts:
        key=cohort_key(spec);candidate,cp=load_cohort(root,key);reference,rp=load_cohort(reference_root,key)
        assert cp['input_sha256']==rp['input_sha256'] and cp['identity']['reference_prediction_sha256']==rp['prediction_file_sha256']
        for name in ('synthetic_seed_index','control','tau'):np.testing.assert_array_equal(candidate[name],reference[name])
        models={'candidate':candidate['prediction'],'expert':candidate['expert']}
        models.update({name:reference[name] for name in ('BL80','B2','D2') if name in reference})
        tau=candidate['tau'];ids=candidate['synthetic_seed_index'];cohort_sources.append({'cohort':key,'candidate_sha256':cp['prediction_file_sha256'],'reference_sha256':rp['prediction_file_sha256']})
        if spec['kind']=='calibration':
            for name,p in models.items():
                maxima=p.max(axis=1)
                for alpha in menu['threshold_alphas']:
                    value=threshold(maxima,alpha);cutoffs[(name,spec['scenario'],alpha)]=value
                    calibration_rows.append({'candidate':cid,'scope':args.scope,'fold':args.fold,'background':spec['scenario'],'model':name,
                        'alpha':alpha,'threshold':value,'independent_null_series':len(maxima),'calibration_horizon':spec['horizon']})
        else:
            horizons=menu['long_null_horizons'] if spec['kind']=='long_null' else [menu['counterexample_horizon']]
            background=backgrounds[scenario_background(spec['scenario'],lock)]
            for horizon in horizons:
                if spec['kind']=='counterexample':
                    for name,p in models.items():point_scores.append({'candidate':cid,'scope':args.scope,'fold':args.fold,'scenario':spec['scenario'],
                        'model':name,'horizon':horizon,'TS_AUC':score(p,tau,ids,horizon),'curves':len(tau),'paired_seeds':counts.shape[1]})
                    progress('Synthetic paired TS-AUC',cid+' '+args.scope+' fold '+str(args.fold)+' '+spec['scenario'])
                    new=score_draws(models['candidate'],tau,ids,horizon,counts);path=out/(spec['scenario']+'_CANDIDATE_TS_AUC_DRAWS.npy')
                    np.save(path,new,allow_pickle=False);files[path.name]=digest(path)
                    for name in ('BL80','B2','D2'):
                        if name not in models:continue
                        old=reference_score_draws(models[name],tau,ids,horizon,counts,scope=args.scope,fold=args.fold,scenario=spec['scenario'],
                            role=name,record=rp,counts_sha=seed_record['sha256'],execution_sha=execution_sha)
                        paired_scores.append({'candidate':cid,'scope':args.scope,'fold':args.fold,'scenario':spec['scenario'],'reference':name,
                            'horizon':horizon,'candidate_TS_AUC':float(new[0]),'reference_TS_AUC':float(old[0]),**paired_summary(old[:,None],new[:,None])})
                statistics={};quantile_columns={}
                for name,p in models.items():
                    quantiles=np.quantile(p[:,:horizon],menu['score_quantiles'])
                    quantile_columns[name]={'score_quantile_'+str(q):float(value) for q,value in zip(menu['score_quantiles'],quantiles)}
                    statistics[name]={'score_q99':float(quantiles[menu['score_quantiles'].index(.99)]),'mean_max_score':float(p[:,:horizon].max(axis=1).mean())}
                for alpha in menu['threshold_alphas']:
                    for name,p in models.items():
                        report=alarm_summary(p[:,:horizon],tau,cutoffs[(name,background,alpha)],score_statistics=statistics[name])
                        report={'candidate':cid,'scope':args.scope,'fold':args.fold,'scenario':spec['scenario'],'model':name,'horizon':horizon,'alpha':alpha,**report}
                        report.update(quantile_columns[name])
                        (long_alarms if spec['kind']=='long_null' else counter_alarms).append(report)
                    events=('null_eventual',) if spec['kind']=='long_null' else ('null_eventual','prebreak','postbreak_detection')
                    for event in events:
                        new=event_draws(models['candidate'][:,:horizon],tau,ids,cutoffs[('candidate',background,alpha)],event,counts)
                        if new is None:continue
                        for name in ('BL80','B2','D2'):
                            if name not in models:continue
                            old=event_draws(models[name][:,:horizon],tau,ids,cutoffs[(name,background,alpha)],event,counts)
                            if old is None:continue
                            alarm_comparisons.append({'candidate':cid,'scope':args.scope,'fold':args.fold,'scenario':spec['scenario'],'reference':name,
                                'horizon':horizon,'alpha':alpha,'event':event,'positive_delta_is_better':event=='postbreak_detection',
                                **paired_summary(old[:,None],new[:,None])})
            if spec['scenario'] in ('O1','OB16'):
                tables=impulse_tables(candidate,reference,candidate_id=cid,scope=args.scope,fold=args.fold,scenario=spec['scenario'],
                    menu=menu,lock=lock,counts=counts,feature_names=model.names)
                impulse_lags.extend(tables[0]);impulse_summary.extend(tables[1]);impulse_comparisons.extend(tables[2]);impulse_features.extend(tables[3])
        progress('Synthetic fixed-cohort analysis complete',cid+' '+args.scope+' fold '+str(args.fold)+' '+key)
        print('NEXT5 SYNTHETIC ANALYSIS',cid,args.scope,args.fold,key,flush=True)
    tables={'CALIBRATION.csv':calibration_rows,'COUNTEREXAMPLE_SCORES.csv':point_scores,'PAIRED_COUNTEREXAMPLES.csv':paired_scores,
        'COUNTEREXAMPLE_ALARMS.csv':counter_alarms,'LONG_NULL_RESULTS.csv':long_alarms,'PAIRED_ALARMS.csv':alarm_comparisons,
        'OUTLIER_LAG_RESPONSE.csv':impulse_lags,'OUTLIER_RESPONSE_SUMMARY.csv':impulse_summary,
        'PAIRED_OUTLIER_RESPONSE.csv':impulse_comparisons,'FEATURE_IMPULSE_RESPONSE.csv':impulse_features}
    for name,rows in tables.items():write_table(out/name,rows);files[name]=digest(out/name)
    model.validate();guard.check_state()
    atomic_json(out/'RESULTS.json',{'created_utc':utc_now(),'status':'PASS','candidate':cid,'scope':args.scope,'fold':args.fold,'new_fits':0,
        'real_Shadow_rows':0,'LOCAL_SEAL_rows':0,'cohorts':len(cohort_sources),'cohort_sources':cohort_sources,'model_proof':model_proof,
        'replicates':2000,'seed_counts_sha256':seed_record['sha256'],'execution_sha256':execution_sha,'files_sha256':files,
        'uncertainty':'Conditional on fitted model and calibration thresholds. Pairing is by generating seed, not independent points or overlapping fitted folds.',
        'no_anytime_alpha_guarantee':True,'synthetic_only_does_not_relax_primary_gate':True,
        'recovery_rule':'After final impulse, first 16 consecutive points with absolute paired score excess <=0.01; censor at horizon2048.'},immutable=True)
    progress('Fitted-fold synthetic report complete',cid+' '+args.scope+' fold '+str(args.fold)+'; no new fits or generator changes')


if __name__=='__main__':main()
