"""Fixed paired all-time/time/age/H/mechanism OOF fractures, no new fits."""
from pathlib import Path
import argparse
import csv
import hashlib
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from numba import set_num_threads
from src.next5.artifacts import BASE,atomic_json,read_json,digest,utc_now,progress
from src.next5.data import prepare_research,admitted_npz
from src.next5.bootstrap import PreparedAUC,paired_summary
from src.next5.fractures import masks,pair_mass
from src.next5.scheduler import register_job
from src.scoring.ts_auc import ts_auc


def write_table(path,rows):
    with path.open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate');parser.add_argument('--register-policy',action='store_true');args=parser.parse_args()
    policy_path=BASE/'diagnostics/FRACTURE_POLICY.json'
    sources=('scripts/forensic_next5_candidate.py','src/next5/fractures.py','src/next5/bootstrap.py','src/next/bootstrap.py')
    if args.register_policy:
        atomic_json(policy_path,{'created_utc':utc_now(),'status':'FROZEN_BEFORE_FRACTURE_SCORES',
            'source_sha256':{n:digest(ROOT/n) for n in sources},'replicates':2000,'new_draws':False,
            'counts_lock_sha256':digest(BASE/'bootstrap/COUNTS_LOCK.json'),'diagnostic_lock_sha256':digest(BASE/'diagnostics/RESEARCH_DIAGNOSTIC_LOCK.json'),
            'supported_positive_series_min':100,'supported_H_null_series_min':100,
            'age_pair_definition':'Positive points restricted to break-age bin; all negative comparators retained at identical online time.',
            'time_age_contributions':'Per-fold scope pair mass / per-fold all-time pair mass times scope delta; averages sum to overall mean fold delta.',
            'mechanism_subsets_nonadditive':True,'conditional_fixed_OOF_not_selection_adjusted':True,'multiple_slices_not_independent_tests':True,
            'posthoc_mechanisms_never_training_features':True,'new_fits':0},immutable=True)
        register_job({'id':'DIAG003_FRACTURE_CONTRACT','kind':'validation','priority':35,'depends_on_success':['DIAG001_RESEARCH_DESCRIPTORS'],
            'argv':['-m','pytest','-q','tests/test_next5_fractures.py','--junitxml=artifacts/next5/diagnostics/FRACTURE_CONTRACTS.xml'],
            'description':'Exact conservation of official time/age pair contributions and mechanism-control masks','retry_safe':False})
        return
    cid=args.candidate
    if not cid:raise ValueError('Candidate required')
    source=BASE/'experiments'/cid/'full';out=BASE/'forensic'/cid
    if (source/'SKIPPED.json').exists():
        atomic_json(out/'SKIPPED.json',{'created_utc':utc_now(),'candidate':cid,'reason':'Full candidate closed by frozen screen','new_fits':0},immutable=True);return
    policy=read_json(policy_path)
    for name,sha in policy['source_sha256'].items():assert digest(ROOT/name)==sha
    result=read_json(source/'RESULTS.json');assert result['status']=='COMPLETE';bootdir=BASE/'bootstrap'/cid;boot=read_json(bootdir/'RESULTS.json')
    out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'START.json',{'created_utc':utc_now(),'candidate':cid,'full_result_sha256':digest(source/'RESULTS.json'),
        'policy_sha256':digest(policy_path),'new_fits':0},immutable=True)
    set_num_threads(2);data=prepare_research();metadata=data.rows();diagnostic=BASE/'diagnostics'
    diagnostic_lock=read_json(diagnostic/'RESEARCH_DIAGNOSTIC_LOCK.json');sigpath=diagnostic/'RESEARCH_SIGNATURES.npz'
    assert digest(sigpath)==diagnostic_lock['files_sha256'][sigpath.name]
    series=admitted_npz(sigpath,data.guard,'ResearchPool evaluation-only mechanism metadata');sort=np.argsort(series['dataset_id'])
    series={k:v[sort] for k,v in series.items()};count_lock=read_json(BASE/'bootstrap/COUNTS_LOCK.json')
    new_by_scope={};old_by_scope={};fold_rows=[];support_rows=[];files={}
    for fold in range(5):
        directory=source/('fold_'+str(fold));receipt=read_json(directory/'RESULTS.json');path=directory/'predictions.npz'
        assert digest(path)==receipt['predictions_file_sha256'];oof=admitted_npz(path,data.guard,'ResearchPool fixed-scope OOF')
        rows=oof['global_row'];np.testing.assert_array_equal(rows,np.flatnonzero(metadata['fold']==fold));ids=metadata['dataset_id'][rows]
        np.testing.assert_array_equal(ids,oof['dataset_id']);index=np.searchsorted(series['dataset_id'],ids);np.testing.assert_array_equal(series['dataset_id'][index],ids)
        target=metadata['target'][rows];times=metadata['time_online'][rows];tau=series['tau'][index]
        scopes=masks(target,times,tau,series['historical_regime'][index],series['signature'][index],series['observed_scale_direction'][index])
        count_receipt=count_lock['folds'][fold];count_path=ROOT/count_receipt['path'];assert digest(count_path)==count_receipt['sha256']
        draws=admitted_npz(count_path,data.guard,'Fixed fracture bootstrap counts');counts=draws['counts'];code=np.searchsorted(draws['dataset_id'],ids).astype(np.int32)
        np.testing.assert_array_equal(draws['dataset_id'][code],ids);full_mass=pair_mass(target,times)
        fold_contribution={'time_':0.,'age_':0.};fold_masses={'time_':0,'age_':0}
        for name,mask in scopes.items():
            mass=pair_mass(target[mask],times[mask]);score=float(ts_auc(target[mask],oof['prediction'][mask],times[mask]))
            baseline=float(ts_auc(target[mask],oof['baseline'][mask],times[mask]));delta=score-baseline
            positive_series=len(np.unique(ids[mask&(target==1)]));null_series=len(np.unique(ids[mask&(tau<0)]))
            support_rows.append({'scope':name,'fold':fold,'points':int(mask.sum()),'positive_series':positive_series,
                'break_series_present':len(np.unique(ids[mask&(tau>=0)])),'null_series':null_series,'pairs':mass})
            contribution=mass/full_mass*delta
            fold_rows.append({'candidate':cid,'scope':name,'fold':fold,'TS_AUC':score,'BL80_TS_AUC':baseline,'delta':delta,
                'pairs':mass,'all_time_pairs':full_mass,'pair_mass_share':mass/full_mass,'weighted_contribution_delta':contribution})
            for prefix in fold_contribution:
                if name.startswith(prefix):fold_contribution[prefix]+=contribution;fold_masses[prefix]+=mass
            if name=='all_time':
                path=bootdir/('fold_'+str(fold)+'.npy');assert digest(path)==boot['files_sha256'][path.name];new=np.load(path,allow_pickle=False)
                refdir=BASE/'bootstrap/ResearchPool_BL80';path=refdir/('fold_'+str(fold)+'.npy');assert digest(path)==read_json(refdir/('fold_'+str(fold)+'.json'))['sha256'];old=np.load(path,allow_pickle=False)
            else:
                refdir=BASE/'forensic/ResearchPool_BL80';refdir.mkdir(parents=True,exist_ok=True)
                filename='fold_'+str(fold)+'_'+name+'.npy';refpath=refdir/filename;refreceipt=refdir/(filename+'.json')
                identity={'scope':name,'fold':fold,'counts_sha256':count_receipt['sha256'],'rows_sha256':hashlib.sha256(rows[mask].tobytes()).hexdigest(),
                    'policy_sha256':digest(policy_path),'baseline_prediction_sha256':hashlib.sha256(oof['baseline'][mask].tobytes()).hexdigest()}
                progress('Fixed paired ResearchPool fracture',cid+' fold '+str(fold)+' '+name+'; same frozen counts')
                if refpath.exists():
                    prior=read_json(refreceipt);assert prior['identity']==identity and digest(refpath)==prior['sha256'];old=np.load(refpath,allow_pickle=False)
                else:
                    old=PreparedAUC.create(target[mask],oof['baseline'][mask],times[mask],code[mask],counts.shape[1]).evaluate(counts) if mask.any() else np.full(len(counts),.5)
                    np.save(refpath,old,allow_pickle=False);atomic_json(refreceipt,{'created_utc':utc_now(),'identity':identity,'sha256':digest(refpath),'new_fits':0},immutable=True)
                new=PreparedAUC.create(target[mask],oof['prediction'][mask],times[mask],code[mask],counts.shape[1]).evaluate(counts) if mask.any() else np.full(len(counts),.5)
                path=out/filename;np.save(path,new,allow_pickle=False);files[path.name]=digest(path)
            assert abs(new[0]-score)<1e-14 and abs(old[0]-baseline)<1e-14
            new_by_scope.setdefault(name,[]).append(new);old_by_scope.setdefault(name,[]).append(old)
        for prefix in fold_contribution:
            assert fold_masses[prefix]==full_mass and abs(fold_contribution[prefix]-receipt['delta'])<1e-12
        print('NEXT5 PAIRED FRACTURES',cid,fold,flush=True)
    summary=[]
    for name in new_by_scope:
        new=np.asarray(new_by_scope[name]).T;old=np.asarray(old_by_scope[name]).T;support=[r for r in support_rows if r['scope']==name]
        positive=sum(r['positive_series'] for r in support);nulls=sum(r['null_series'] for r in support)
        supported=positive>=policy['supported_positive_series_min'] and (not name.startswith('regime_') or nulls>=policy['supported_H_null_series_min'])
        summary.append({'candidate':cid,'scope':name,'candidate_mean':float(new[0].mean()),'baseline_mean':float(old[0].mean()),
            'positive_series':positive,'null_series':nulls,'points':sum(r['points'] for r in support),'supported':supported,
            'mean_weighted_contribution_delta':float(np.mean([r['weighted_contribution_delta'] for r in fold_rows if r['scope']==name])),**paired_summary(old,new)})
    for name,table in (('PAIRED_FRACTURES.csv',summary),('FOLD_SCORES.csv',fold_rows),('SUPPORT.csv',support_rows)):
        write_table(out/name,table);files[name]=digest(out/name)
    gate=read_json(ROOT/'configs/next5/RESEARCH_POLICY_LOCK.json')['catastrophic_gate']
    catastrophic=[r for r in summary if r['supported'] and ((r['scope'].startswith('mechanism_') and r['estimate_mean_delta']<gate['supported_mechanism_delta_strict_below']) or
        (r['scope'].startswith('regime_') and r['estimate_mean_delta']<gate['supported_H_regime_delta_strict_below']))]
    atomic_json(out/'RESULTS.json',{'created_utc':utc_now(),'status':'PASS','candidate':cid,'scopes':len(summary),'new_fits':0,
        'shadow_rows':0,'seal_rows':0,'replicates':2000,'new_random_draws':False,'files_sha256':files,
        'time_and_age_pair_mass_and_delta_conservation':True,'supported_catastrophic_slices':catastrophic,
        'H_regime_scope':'Conditional original ResearchPool OOF subgroup; separate H-exclusion refits required for that claim.',
        'conditional_fixed_OOF_not_selection_adjusted':True,'multiple_scopes_not_independent_tests':True,
        'source_result_sha256':digest(source/'RESULTS.json'),'policy_sha256':digest(policy_path),'selection_authority':'Fixed diagnostic gates only; no recipe tuning'},immutable=True)
    data.guard.check_state();progress('Paired ResearchPool fractures complete',cid+'; fixed scope support, time/age exact contribution conservation and no extra draws')


if __name__=='__main__':main()
