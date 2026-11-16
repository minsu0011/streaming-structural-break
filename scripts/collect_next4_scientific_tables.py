"""Join immutable model evidence into reviewable tables without new scoring."""
import argparse
from pathlib import Path
import sys
import re
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True)
    parser.add_argument('--require-complete-synthetic',action='store_true');args=parser.parse_args()
    if not re.fullmatch('[A-Za-z0-9_]+',args.tag):raise ValueError('Safe immutable snapshot name required')
    out=BASE/'report_tables'/args.tag;out.mkdir(parents=True,exist_ok=False)
    sources={}
    def read(path):
        sources[path.relative_to(ROOT).as_posix()]=digest(path)
        return read_json(path)
    def csv(path):
        sources[path.relative_to(ROOT).as_posix()]=digest(path)
        return pd.read_csv(path)
    def role(candidate):
        if candidate.startswith(('B1_','B2_','G1_')):return 'augmented_J102'
        if candidate.startswith(('E1_','E2_','G2_')):return 'learned_combination_of_existing_experts'
        return 'independent_state_readout' if candidate.startswith('H1_') else 'standalone_new_expert'
    models=[];cv=[];bootstraps=[];fractures=[];diversity=[];alts=[];regimes=[];counter=[];longs=[]
    records={};screen_count=full_count=selection_fits=meta_fits=alternate_fits=0
    support=csv(BASE/'scored_support/SCORED_SUPPORT.csv')
    for path in sorted((BASE/'registrations').glob('*.json')):
        reg=read(path);cid=reg['experiment_id'];record={'candidate':cid,'family':reg['family'],
            'hypothesis':reg['hypothesis'],'expert_role':role(cid),'registered_utc':reg['created_utc'],
            'created_before_score':reg['created_before_score'],'registration_sha256':digest(path)}
        stages={}
        for scope in ('screen','full'):
            result_path=BASE/'experiments'/cid/scope/'RESULTS.json'
            if not result_path.exists():continue
            r=read(result_path);assert r['status']=='COMPLETE' and r['registration_sha256']==digest(path)
            stages[scope]=r;selection_fits+=r['new_fits'];meta_fits+=r.get('new_meta_fits',0)
            if scope=='screen':screen_count+=1
            else:full_count+=1
            record.update({scope+'_'+key:r.get(key) for key in ('mean','mean_delta','median','worst_fold','median_delta','worst_delta','standalone_mean','new_fits','disposition')})
            record[scope+'_feature_count']=r['feature_count']
            for f in r['fold_records']:
                cv.append({'candidate':cid,'family':reg['family'],'scope':scope,'fold':f['fold'],
                    'ts_auc':f['ts_auc'],'reference_ts_auc':f['reference_ts_auc'],'delta':f['delta'],
                    'standalone_ts_auc':f['standalone_ts_auc'],'new_fits':f['new_fits'],
                    'frozen_base_new_fits':f['frozen_base_new_fits'],
                    'training_series':f['training'].get('training_series'),
                    'validation_series':f['training'].get('validation_series'),
                    'training_rows':f['training'].get('training_rows'),
                    'validation_rows':f['training'].get('validation_rows'),
                    'expert_role':role(cid)})
        record['evaluation_status']='FULL_COMPLETED' if 'full' in stages else 'SCREEN_ONLY'
        record['performance_primary_and_bootstrap_admission']=False
        record['robustness_primary_scale_admission']=False
        bp=BASE/'bootstrap'/cid/'RESULTS.json'
        if bp.exists():
            b=read(bp);assert b['status']=='PASS'
            record['performance_primary_and_bootstrap_admission']=b['performance_primary_and_bootstrap_admission']
            record.update({'primary_'+key:b['paired_all_time'][key] for key in ('ci_2_5','ci_97_5','probability_delta_positive')})
            bootstraps.append({'candidate':cid,'family':reg['family'],'reference':'BL2_C205_DEPENDENCE_80',
                'scope':'original_primary_five_fold','unit':b['unit'],'folds':b['folds'],
                **b['paired_all_time'],'performance_primary_and_bootstrap_admission':b['performance_primary_and_bootstrap_admission'],
                'conditional_fitted_OOF_not_selection_adjusted':True})
        fp=BASE/'forensic'/cid/'RESULTS.json'
        if fp.exists():
            f=read(fp);assert f['status']=='PASS'
            record['robustness_primary_scale_admission']=f['robustness_primary_scale_admission']
            record.update({'SCALE_'+key:f['paired_scale'][key] for key in ('estimate_mean_delta','ci_2_5','ci_97_5','break_series')})
            fractures.append(csv(fp.parent/'PAIRED_FRACTURES.csv'))
            frame=csv(fp.parent/'STANDALONE_DIVERSITY.csv');frame['scope']='full';frame['expert_role']=role(cid);diversity.append(frame)
        rp=BASE/'robustness'/cid/'RESULTS.json'
        if rp.exists():
            r=read(rp);assert r['status']=='PASS';alternate_fits+=r['new_fits']
            record['ALT_positive_count']=r['ALT_positive_count'];record['H_minimum_matched_delta']=r['H_minimum_matched_delta']
            for split in r['summaries']:
                s={'candidate':cid,'family':reg['family'],'split':split['split'],
                    'mean':split['mean'],'mean_delta':split['mean_delta'],'median':split['median'],'worst':split['worst'],
                    'minimum_matched_fold_delta':split['minimum_matched_fold_delta'],'new_fits':split['new_fits'],
                    **split['paired_all_time'],'scope_note':'Existing DEV under frozen alternative grouped partition; not independent new data'}
                if split['split'].startswith('ALT_'):alts.append(s)
                else:
                    for fold,(a,b) in enumerate(zip(split['fold_scores'],split['reference_fold_scores'])):
                        regimes.append({'candidate':cid,'record_type':'HELD_OUT_H_REGIME_REFIT','regime':fold,
                            'ts_auc':a,'reference_ts_auc':b,'delta':a-b,'validation_unit':'entire excluded H regime',
                            'all_regime_mean_delta':split['mean_delta'],'all_regime_ci_lower':split['paired_all_time']['ci_2_5'],
                            'all_regime_ci_upper':split['paired_all_time']['ci_97_5']})
        models.append(record);records[cid]=stages
    assert screen_count==16 and full_count==14 and selection_fits==102 and meta_fits==16
    frames=pd.concat(fractures,ignore_index=True)
    # Membership counts are preserved, but scoring-positive support is explicit.
    support_columns=['scope','positive_series','negative_series','eligible_positive_series','eligible_negative_series',
        'eligible_times','positive_negative_pair_mass','at_least_100_scoring_positive_series']
    frames=frames.merge(support[support_columns],on='scope',how='left',validate='many_to_one')
    assert frames['positive_series'].notna().all()
    for row in frames[frames.scope.str.startswith('regime_')].to_dict('records'):
        regimes.append({'candidate':row['candidate'],'record_type':'ORIGINAL_OOF_CONDITIONAL_H_SLICE',
            'regime':int(row['scope'].split('_')[1]),'ts_auc':row['candidate_mean'],'reference_ts_auc':row['reference_mean'],
            'delta':row['estimate_mean_delta'],'ci_lower':row['ci_2_5'],'ci_upper':row['ci_97_5'],
            'positive_series':row['positive_series'],'validation_unit':'slice of original OOF; H regime not excluded during fit'})
    closed=csv(BASE/'closed_menu_diagnostics/DIVERSITY.csv')
    closed['expert_role']=closed.candidate.map(role)
    diversity=pd.concat([*diversity,closed],ignore_index=True).drop_duplicates(['candidate','scope','fold','reference'],keep='last')
    numeric=[c for c in diversity if c not in ('candidate','scope','fold','expert_role','reference')]
    complements=diversity.groupby(['candidate','scope','expert_role','reference'],as_index=False)[numeric].mean()
    complements['fold_aggregation']='unweighted mean of existing validation-fold diagnostics'
    model_frame=pd.DataFrame(models)
    complements=complements.merge(model_frame[['candidate','screen_standalone_mean','full_standalone_mean','full_mean_delta']],on='candidate',validate='many_to_one')
    synthetic_complete=[];synthetic_pending=[]
    for npth in sorted((BASE/'synthetic_nominations').glob('*.json')):
        nomination=read(npth);p=BASE/'synthetic/candidates'/nomination['id']/'RESULTS.json'
        if not p.exists():synthetic_pending.append(nomination['id']);continue
        r=read(p);assert r['status']=='PASS';synthetic_complete.append(nomination['id'])
        counter.append(csv(p.parent/'COUNTEREXAMPLES.csv'));longs.append(csv(p.parent/'LONG_NULL.csv'))
    if args.require_complete_synthetic:assert not synthetic_pending
    ranking=csv(BASE/'closed_menu_diagnostics/PAIRWISE_FIT_STRUCTURE.csv')
    tables={'MODEL_RESULTS_NEXT4.csv':model_frame,'CV_RESULTS_NEXT4.csv':pd.DataFrame(cv),
        'PAIRED_BOOTSTRAP_NEXT4.csv':pd.DataFrame(bootstraps),'COMPLEMENT_RESULTS_NEXT4.csv':complements,
        'STANDALONE_DIVERSITY_FOLDS_NEXT4.csv':diversity,'ALT_RESULTS_NEXT4.csv':pd.DataFrame(alts),
        'REGIME_RESULTS_NEXT4.csv':pd.DataFrame(regimes),
        'MECHANISM_RESULTS_NEXT4.csv':frames[frames.scope.str.startswith('mechanism_')],
        'TIME_FRACTURE_NEXT4.csv':frames[frames.scope.str.startswith(('time_','age_'))],
        'COUNTEREXAMPLE_RESULTS_NEXT4.csv':pd.concat(counter,ignore_index=True) if counter else pd.DataFrame(columns=['candidate','nomination','training_scope','scenario','alpha','ts_auc','reference_ts_auc','delta']),
        'LONG_NULL_RESULTS_NEXT4.csv':pd.concat(longs,ignore_index=True) if longs else pd.DataFrame(columns=['candidate','nomination','training_scope','regime','horizon','alpha','eventual_alarm_rate','reference_eventual_alarm_rate']),
        'RANKING_OBJECTIVE_NEXT4.csv':ranking[ranking.candidate.str.startswith('E')],
        'H_CONTEXT_RESULTS.csv':model_frame[model_frame.family=='G'],
        'H_CONTEXT_FITTED_STRUCTURE_NEXT4.csv':csv(BASE/'closed_menu_diagnostics/G1_CONTEXT_USE.csv'),
        'TEMPORAL_STATE_RESULTS_NEXT4.csv':model_frame[model_frame.family=='H']}
    for family,name in [('A','DEPENDENCE_V2_RESULTS.csv'),('B','SCALE_ASYMMETRY_RESULTS.csv'),
            ('C','DISTRIBUTION_SHIFT_RESULTS.csv'),('D','SEQUENTIAL_LIKELIHOOD_RESULTS.csv')]:
        tables[name]=model_frame[model_frame.family==family]
    for name,frame in tables.items():frame.to_csv(out/name,index=False)
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'tag':args.tag,
        'research_or_audit_snapshot_only':True,'does_not_claim_final_audit_or_promotion':True,
        'screen_candidate_count':screen_count,'full_candidate_count':full_count,
        'selection_learner_fits':selection_fits,'selection_meta_fits':meta_fits,
        'selection_base_fits':selection_fits-meta_fits,'completed_ALT_H_learner_fits':alternate_fits,
        'synthetic_complete':synthetic_complete,'synthetic_pending':synthetic_pending,
        'new_fits_in_this_collection':0,'new_scores_in_this_collection':0,'seal_rows':0,
        'support_membership_and_scoring_positive_counts_separated':True,
        'original_conditional_H_slices_and_heldout_H_refits_separated':True,
        'input_sha256':sources,'source_sha256':digest(Path(__file__)),
        'table_rows':{name:len(frame) for name,frame in tables.items()},
        'files_sha256':{name:digest(out/name) for name in tables}},immutable=True)
    print('NEXT4 SCIENTIFIC TABLE SNAPSHOT',args.tag,len(tables),'synthetic pending',len(synthetic_pending),flush=True)


if __name__=='__main__':main()
