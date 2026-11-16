"""Freeze executable stress protocol and equal-size seed pairs before replay."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.next5.artifacts import BASE,atomic_json,read_json,digest,utc_now,progress
from src.next5.scheduler import register_job
from src.validation.group_bootstrap import draw_group_multiplicities


def main():
    test=BASE/'synthetic/EXECUTION_CONTRACTS.xml'
    state=read_json(BASE/'jobs/SYN001_EXECUTION_CONTRACTS/STATE.json')
    assert state['status']=='COMPLETE' and state['exit_code']==0
    tree=ET.parse(test);assert not tree.findall('.//failure') and not tree.findall('.//error')
    assert not list((BASE/'synthetic').glob('**/REPLAY_START.json'))
    menu=read_json(BASE/'synthetic/MENU_LOCK.json');sizes={}
    for n in sorted({menu['fold_model_sizes']['evaluation'],menu['frozen_whole_ResearchPool_model_sizes']['evaluation']}):
        ids=np.arange(n,dtype=np.int32);code,counts=draw_group_multiplicities(ids,2000,20260908)
        np.testing.assert_array_equal(code,ids);assert np.all(counts.sum(axis=1)==n) and np.all(counts[0]==1)
        path=BASE/'synthetic'/('PAIRED_SEED_COUNTS_'+str(n)+'.npz')
        with path.open('xb') as stream:np.savez(stream,synthetic_seed_index=ids,counts=counts)
        sizes[str(n)]={'path':path.relative_to(ROOT).as_posix(),'sha256':digest(path),'replicates':2000,'seed':20260908,'unit':'paired generating seed'}
    atomic_json(BASE/'synthetic/SEED_COUNTS_LOCK.json',{'created_utc':utc_now(),'sizes':sizes,
        'same_counts_across_equal_size_scenarios_and_fitted_models':True,'not_real_data_bootstrap':True},immutable=True)
    sources=('scripts/predict_next5_synthetic.py','scripts/analyze_next5_synthetic.py',
        'src/next5/synthetic.py','src/next5/synthetic_cache.py','src/next5/synthetic_metrics.py',
        'src/next5/impulse_reports.py','src/next5/bootstrap.py','src/next/bootstrap.py',
        'src/validation/group_bootstrap.py','src/next2/counterexamples.py','src/next2/cross_background_scale.py',
        'src/scoring/ts_auc.py','src/next5/model.py','src/next5/guard.py')
    atomic_json(BASE/'synthetic/EXECUTION_LOCK.json',{'created_utc':utc_now(),'status':'FROZEN_BEFORE_FITTED_SYNTHETIC_REPLAY',
        'menu_sha256':digest(BASE/'synthetic/MENU_LOCK.json'),'seed_counts_lock_sha256':digest(BASE/'synthetic/SEED_COUNTS_LOCK.json'),
        'sources_sha256':{name:digest(ROOT/name) for name in sources},'contract_tests_sha256':digest(test),
        'score_precision':'float64, exact ties; no silent float32 prediction conversion',
        'calibration_uncertainty':'Thresholds fixed from independent partition-0 nulls; reported intervals condition on those thresholds and fitted artifacts.',
        'recovery_rule':{'after_final_impulse':True,'absolute_paired_score_excess_tolerance':.01,'consecutive_points':16,'censor_horizon':2048},
        'fold_dependence':'Five fitted folds overlap in training; do not treat them as independent bootstrap units.',
        'reference_bootstrap_cache':'Exact fitted scope/fold/model, input prediction bytes, execution and count hashes required.',
        'coverage':menu['candidate_coverage'],'no_new_fits':True,'primary_gate_not_relaxed':True},immutable=True)
    for scope,folds in (('full',range(5)),('screen',(0,4))):
        for fold in folds:
            dependencies=['SYN001_EXECUTION_CONTRACTS','REF_J102_'+scope,'REF_C205_'+scope]
            if scope=='full':dependencies+=['REF_B2_augmented_J_full','REF_D2_expert_full']
            register_job({'id':f'SYN_REF_{scope}_{fold}','kind':'validation','priority':76 if scope=='full' and fold==0 else 85,
                'depends_on_success':dependencies,'argv':['scripts/predict_next5_synthetic.py','--reference','--scope',scope,'--fold',str(fold)],
                'description':'Matched ResearchPool fitted reference replay across the unchanged 29-cohort menu. '+scope+' fold '+str(fold),
                'retry_safe':True})
    completed=[]
    for path in sorted((BASE/'registrations').glob('*.json')):
        cid=path.stem;full=BASE/'experiments'/cid/'full/RESULTS.json'
        scope='full' if full.exists() else 'screen'
        if scope=='screen':assert read_json(BASE/'experiments'/cid/'screen/RESULTS.json')['screen_pass'] is False
        result=read_json(BASE/'experiments'/cid/scope/'RESULTS.json')
        completed.append((cid,scope,result['mean_delta']))
    assert len(completed)==20
    completed.sort(key=lambda row:(-row[2],row[0]))
    coverage=[]
    for rank,(cid,scope,delta) in enumerate(completed):
        for fold in (range(5) if scope=='full' else (0,4)):
            job=f'SYN_{cid}_{scope}_{fold}'
            priority=83 if scope=='full' and fold==0 else 90
            register_job({'id':job,'kind':'validation','priority':priority,'depends_on_success':[f'SYN_REF_{scope}_{fold}',cid+'_'+scope],
                'argv':['scripts/predict_next5_synthetic.py','--candidate',cid,'--scope',scope,'--fold',str(fold)],
                'description':'Complete frozen synthetic menu for an already fitted artifact; no new fit. '+cid+' '+scope+' fold '+str(fold),
                'retry_safe':True})
            register_job({'id':'SYN_ANALYZE_'+cid+'_'+scope+'_'+str(fold),'kind':'validation','priority':priority,
                'depends_on_success':[job],
                'argv':['scripts/analyze_next5_synthetic.py','--candidate',cid,'--scope',scope,'--fold',str(fold)],
                'description':'Fixed 2000 paired-seed bootstrap, alarms, long-null and impulse response. '+cid+' '+scope+' fold '+str(fold),
                'retry_safe':False})
            coverage.append({'candidate':cid,'scope':scope,'fold':fold,'model_sha256':read_json(BASE/'experiments'/cid/scope/('fold_'+str(fold))/'RESULTS.json')['model_sha256']})
    atomic_json(BASE/'synthetic/FITTED_COVERAGE_LOCK.json',{'created_utc':utc_now(),'fitted_artifacts':coverage,
        'artifact_count':len(coverage),'candidate_count':20,'new_fits':0,'failed_screen_stays_failed':True,
        'execution_lock_sha256':digest(BASE/'synthetic/EXECUTION_LOCK.json')},immutable=True)
    progress('Synthetic execution and fitted coverage frozen','97 existing fitted candidate artifacts, seven matched reference folds, unchanged seeds/DGPs and exact paired counts; no new fits')
    print('NEXT5 SYNTHETIC EXECUTION FROZEN',len(coverage),'candidate artifacts',flush=True)


if __name__=='__main__':main()
