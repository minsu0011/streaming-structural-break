"""Fixed fitted-OOF fractures and standalone diversity against three references."""
from pathlib import Path
import sys
import argparse
import hashlib
import numpy as np
import pandas as pd
from numba import set_num_threads
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,atomic_json,digest,read_json,utc_now,progress
from src.next4.frozen import fold_predictions
from src.next4.diagnostics import evaluation_masks,scale_directions,diversity,source_hash
from src.next.data import prepare_dev
from src.next.bootstrap import PreparedAUC,paired_summary
from src.validation.group_bootstrap import draw_group_multiplicities
from scripts.forensic_next_candidate import rank_within_time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',required=True);args=parser.parse_args()
    policy_path=ROOT/'configs/next4/RESEARCH_POLICY_LOCK.json';policy=read_json(policy_path)
    lock=BASE/'diagnostic_policy/FORENSIC_POLICY.json';contract=read_json(lock)
    assert contract['source_sha256']=={**source_hash(),'scripts/forensic_next4_candidate.py':digest(Path(__file__))}
    source=BASE/'experiments'/args.candidate/'full';full=read_json(source/'RESULTS.json')
    assert full['status']=='COMPLETE' and full['scope']=='full'
    out=BASE/'forensic'/args.candidate;out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'START.json',{'timestamp':utc_now(),'candidate':args.candidate,'full_result_sha256':digest(source/'RESULTS.json'),
        'diagnostic_policy_sha256':digest(lock),'candidate_registration_sha256':full['registration_sha256'],'new_fits':0},immutable=True)
    set_num_threads(2);data=prepare_dev(ROOT);metadata=data.rows()
    meta_path=ROOT/'artifacts/next/current_forensic/DEV_SIGNATURES.parquet'
    assert digest(meta_path)==read_json(meta_path.parent/'REGIME_ASSIGNMENT_LOCK.json')['files_sha256'][meta_path.name]
    series=pd.read_parquet(meta_path).set_index('dataset_id')
    data.guard.admit(series.index.to_numpy(),purpose='NEXT4 evaluation-only fixed mechanism and H-regime scopes')
    directions,direction_report=scale_directions(data,series)
    new,old,support,fold_scores,diversity_rows={},{},[],[],[]
    for fold in range(5):
        rows=np.flatnonzero(metadata['fold']==fold);record=read_json(source/f'fold_{fold}.json')
        for filename in (f'fold_{fold}.npy',f'fold_{fold}_expert.npy',f'fold_{fold}_global_rows.npy'):
            assert digest(source/filename)==record['files_sha256'][filename]
        np.testing.assert_array_equal(np.load(source/f'fold_{fold}_global_rows.npy',allow_pickle=False),rows)
        p=np.load(source/f'fold_{fold}.npy',allow_pickle=False);expert=np.load(source/f'fold_{fold}_expert.npy',allow_pickle=False)
        ids,y,t=(metadata[k][rows] for k in ('dataset_id','target','time_online'));meta=series.reindex(ids)
        baseline,baseline_sources=fold_predictions(data,fold,'full',rows)
        er=rank_within_time(expert,t)
        for name in ('J102','C205','BL80'):
            diversity_rows.append({'candidate':args.candidate,'expert_role':'augmented_J102' if full['family']=='B' else 'standalone_new_expert',
                'reference':name,'fold':fold,**diversity(expert,baseline[name],t,y,expert_ranks=er)})
        code,counts=draw_group_multiplicities(ids,policy['bootstrap']['draws'],policy['bootstrap']['seed']+fold)
        for scope,mask in evaluation_masks(meta,y,t,directions).items():
            progress('NEXT4 fixed-scope uncertainty',args.candidate+' fold '+str(fold)+' '+scope)
            if scope=='all_time':
                previous=BASE/'bootstrap'/args.candidate;boot=read_json(previous/'RESULTS.json')
                assert digest(previous/f'fold_{fold}.npy')==boot['files_sha256'][f'fold_{fold}.npy']
                candidate_values=np.load(previous/f'fold_{fold}.npy',allow_pickle=False)
                refdir=BASE/'bootstrap/frozen_BL80';refrecord=read_json(refdir/f'fold_{fold}.json')
                assert refrecord['identity']['counts_sha256']==hashlib.sha256(counts.tobytes()).hexdigest()
                assert digest(refdir/f'fold_{fold}.npy')==refrecord['sha256']
                reference_values=np.load(refdir/f'fold_{fold}.npy',allow_pickle=False)
            else:
                filename=f'fold_{fold}_{scope}.npy';refdir=BASE/'forensic/frozen_BL80';refdir.mkdir(parents=True,exist_ok=True)
                refpath=refdir/filename;receipt_path=refdir/(filename+'.json')
                identity={'counts_sha256':hashlib.sha256(counts.tobytes()).hexdigest(),'rows_sha256':hashlib.sha256(rows[mask].tobytes()).hexdigest(),
                    'diagnostic_policy_sha256':digest(lock),'signature_sha256':digest(meta_path),'scope':scope,'fold':fold}
                if refpath.exists():
                    prior=read_json(receipt_path);assert prior['identity']==identity and digest(refpath)==prior['sha256']
                    reference_values=np.load(refpath,allow_pickle=False)
                else:
                    reference_values=PreparedAUC.create(y[mask],baseline['BL80'][mask],t[mask],code[mask],counts.shape[1]).evaluate(counts)
                    np.save(refpath,reference_values,allow_pickle=False)
                    atomic_json(receipt_path,{'timestamp':utc_now(),'identity':identity,'sha256':digest(refpath),'frozen_OOF_sources':baseline_sources},immutable=True)
                candidate_values=PreparedAUC.create(y[mask],p[mask],t[mask],code[mask],counts.shape[1]).evaluate(counts)
                np.save(out/filename,candidate_values,allow_pickle=False)
            new.setdefault(scope,[]).append(candidate_values);old.setdefault(scope,[]).append(reference_values)
            support.append({'scope':scope,'fold':fold,'break_series':len(np.unique(ids[mask&(meta.tau.to_numpy()>=0)])),
                'null_series':len(np.unique(ids[mask&(meta.tau.to_numpy()<0)])),'points':int(mask.sum())})
            fold_scores.append({'candidate':args.candidate,'fold':fold,'scope':scope,'ts_auc':candidate_values[0],
                'reference_ts_auc':reference_values[0],'delta':candidate_values[0]-reference_values[0]})
        print('NEXT4 FORENSIC FOLD',args.candidate,fold,flush=True)
    totals=pd.DataFrame(support).groupby('scope')[['break_series','null_series','points']].sum();reports=[]
    for scope in new:
        a=np.asarray(new[scope]).T;b=np.asarray(old[scope]).T
        reports.append({'candidate':args.candidate,'reference':'BL2_C205_DEPENDENCE_80','scope':scope,
            'candidate_mean':float(a[0].mean()),'reference_mean':float(b[0].mean()),
            'candidate_median':float(np.median(a[0])),'candidate_worst':float(a[0].min()),
            'supported':bool(totals.loc[scope,'break_series']>=100 and (not scope.startswith('regime_') or totals.loc[scope,'null_series']>=100)),
            **totals.loc[scope].astype(int).to_dict(),**paired_summary(b,a)})
    table=pd.DataFrame(reports);table.to_csv(out/'PAIRED_FRACTURES.csv',index=False)
    pd.DataFrame(fold_scores).to_csv(out/'FOLD_SCORES.csv',index=False)
    pd.DataFrame(diversity_rows).to_csv(out/'STANDALONE_DIVERSITY.csv',index=False)
    pd.DataFrame(support).to_csv(out/'SUPPORT.csv',index=False)
    scale=next(r for r in reports if r['scope']=='mechanism_SCALE_DOMINANT');gate=policy['robustness']
    major=[r for r in reports if r['scope'].startswith('mechanism_') and not r['scope'].startswith('mechanism_observed_') and r['supported']]
    admitted=bool(full['mean_delta']>=gate['overall_mean_delta_min'] and scale['supported'] and
        scale['estimate_mean_delta']>=gate['SCALE_delta_min'] and scale['ci_2_5']>0 and
        all(r['estimate_mean_delta']>=gate['minimum_other_supported_mechanism_delta'] for r in major))
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'candidate':args.candidate,'scopes':len(reports),
        'replicates':2000,'new_fits':0,'seal_rows':0,'paired_scale':scale,'robustness_primary_scale_admission':admitted,
        'robustness_remaining_checks':['supported variance-increase damage','ARCH and tail decrease synthetic comparisons','causal/runtime verification'] if admitted else [],
        'directional_scope':direction_report,'H_scope_is_conditional_original_OOF_not_heldout_regime_refit':True,
        'diversity_definitions':contract['diversity_definitions'],'promotion':False,
        'conditional_not_selection_adjusted':True,'diagnostic_policy_sha256':digest(lock),
        'files_sha256':{p.name:digest(p) for p in out.iterdir() if p.is_file()}},immutable=True)
    data.guard.check_state();progress('NEXT4 fixed-scope diagnostics completed',args.candidate+' robustness SCALE admission: '+str(admitted))
    print('NEXT4 FORENSIC COMPLETE',args.candidate,'SCALE delta',scale['estimate_mean_delta'],'SCALE CI',scale['ci_2_5'],scale['ci_97_5'],flush=True)


if __name__=='__main__':main()
