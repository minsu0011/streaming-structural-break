"""Freeze two small honest pairwise readouts before real-data meta fitting."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,atomic_json,read_json,digest,utc_now,progress
from src.next4.pairwise import source_hash
from src.next4.nested_reuse import source_hash as nested_source_hash
from src.next4.scheduler import register_job


def main():
    policy_path=ROOT/'configs/next4/RESEARCH_POLICY_LOCK.json';policy=read_json(policy_path)
    ids=read_json(ROOT/'configs/next4/HYPOTHESIS_FAMILY_LOCK.json')['families']['E']['candidate_ids'];kinds=['linear','interactions']
    xml=BASE/'maintenance/FAMILY_E_FEATURE_TESTS.xml';suites=list(ET.parse(xml).getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==10 and all(int(s.get('failures',0))+int(s.get('errors',0))==0 for s in suites)
    execution=['scripts/evaluate_next4_pairwise.py','src/next4/frozen.py','src/next/data.py','src/next/guard.py','src/scoring/ts_auc.py']
    menu={'status':'FROZEN_BEFORE_FIRST_FAMILY_E_REAL_DATA_SCORE','created_utc':utc_now(),'family':'E','candidate_ids':ids,'kinds':kinds,
        'linear_features':['J102_probability','C205_probability'],
        'interaction_features':['J102_probability','C205_probability','J102_squared','C205_squared','J102_times_C205','absolute_probability_gap'],
        'training_inputs':'Only original within-outer-training inner OOF component probabilities; no outer validation scores or labels in the optimizer',
        'reusable_base_training':'Original 56 inner base fits audited across screen2/full5 outer folds, 4 inner folds each and 2 fixed base recipes',
        'label_objective':'log(1+exp(-(f_positive-f_negative)/0.1)) for two distinct training series at the same online time',
        'pair_sampling':'1048576 pairs per meta fit; eligible time sampled proportional to positive_count * negative_count, then uniform positive and negative rows',
        'pair_seed':20260908,'temperature':.1,'ridge':.01,'ridge_anchor':[.8,.2,0.,0.,0.,0.],
        'linear_coefficient_bounds':[[0,4],[0,4]],'interaction_extra_bounds':[[-4,4]]*4,
        'optimizer':'L-BFGS-B, maxiter 200, ftol 1e-12, gtol 1e-8, maxls 30; no failed-convergence score',
        'intercept':'No intercept fitted: it cancels within pair differences',
        'inference':'Frozen affine/quadratic score map followed by sigmoid(f-0.5), then float32; no learned calibration',
        'deployment_scope':'New combination of existing experts; no claim of an independent new raw signal',
        'screen_gate':{k:policy['screen'][k] for k in ['mean_delta_min','each_fold_delta_min']},
        'stop_rule':'Exactly these two feature maps and fixed optimizer settings; no pair budget, weight, loss-temperature, ridge, score transform or seed search after scores.',
        'novelty':'Prior pairwise experiments trained boosted ranking learners on raw feature designs, while NEXT3 gates used binary/residual objectives. These use a very small constrained pairwise linear/quadratic readout trained on genuinely nested base OOF.',
        'prior_dedup_sources':{p:digest(ROOT/p) for p in ['docs/PAIRWISE_OBJECTIVE_POLICY.md','configs/next_ranking_candidates.json',
            'configs/next_variance_objective_candidates.json','src/next3/learned_router.py','src/next3/router.py']},
        'feature_test_xml_sha256':digest(xml),'feature_tests':'10 cases: 100 current-score prefix cases per map, gradient finite differences, same-time pair-mass/coordinate checks, two synthetic optimizer and serialization tests, invalid inputs',
        'prototype_synthetic_unit_fits_before_registration':2,'new_real_data_meta_fits_before_registration':0,'new_real_model_quality_scores_before_registration':0,
        'model_source_sha256':source_hash(),'nested_source_sha256':nested_source_hash(),'policy_sha256':digest(policy_path),
        'priority_reason':'Independent objective family prioritized in bottleneck review 002; high DEV uncertainty prohibits microvariants and global CASE G still applies'}
    menu_path=BASE/'families/E_MENU_LOCK.json';atomic_json(menu_path,menu,immutable=True)
    for candidate,kind in zip(ids,kinds):
        hyp={'experiment_id':candidate,'parent':'BL2_C205_DEPENDENCE_80','family':'E',
            'hypothesis':'A tiny '+kind+' readout optimized for same-time pairs may rank existing expert evidence better than their frozen 80/20 probability sum.',
            'causal_mechanism':'Pure current-point map of two causally computed scores; coefficients learned only from honest inner base OOF inside outer training.',
            'exact_changes':{'feature_map':kind,'learner':'bounded small pairwise logistic readout','new_raw_features':0},
            'parameters':{'version':'honest_same_time_pairs_v1','kind':kind,'pairs':1048576,'seed':20260908,'temperature':.1,'ridge':.01,'maxiter':200},
            'candidate_menu_id':'E_MENU_V1','expected_gain':'Improve official held-out TS-AUC through metric-aligned finite function fitting with honest meta labels.',
            'risk':'Inner/outer base-fit distribution shift, limited expressivity, surrogate mismatch and adaptive reuse may erase apparent training improvement.',
            'screen_gate':menu['screen_gate'],'full_gate':policy['performance'],'stop_rule':menu['stop_rule'],
            'created_before_score':True,'created_utc':utc_now(),'model_source_sha256':source_hash(),'nested_source_sha256':nested_source_hash(),
            'execution_source_sha256':{p:digest(ROOT/p) for p in execution},'family_menu_sha256':digest(menu_path),
            'policy_sha256':digest(policy_path),'previous_failure_distinction':menu['novelty']}
        atomic_json(BASE/'registrations'/(candidate+'.json'),hyp,immutable=True)
    for index,candidate in enumerate(ids):
        register_job({'id':f'E{index+1:03d}_SCREEN','kind':'research','family':'E','candidate_id':candidate,'priority':50,
            'depends_on_success':['E000_NESTED_REUSE_AUDIT'],'argv':['scripts/evaluate_next4_pairwise.py','--candidate',candidate,'--scope','screen'],
            'description':'Registered same-time pairwise readout on audited honest inner OOF only','retry_safe':False})
    progress('Family E registered','Two finite pairwise maps; 0 real meta scores before registration; required audit of all 56 old inner fits')
    print('NEXT4 FAMILY E FROZEN',len(ids),'screen jobs queued',flush=True)


if __name__=='__main__':main()
