"""Freeze the exact three-candidate correlation-departure family before fitting."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,atomic_json,digest,read_json,utc_now,progress
from src.next4.dependence import implementation_hash,feature_names
from src.next4.model import source_hash
from src.next4.scheduler import register_job


def main():
    policy_path=ROOT/'configs/next4/RESEARCH_POLICY_LOCK.json';policy=read_json(policy_path)
    plan=read_json(ROOT/'configs/next4/HYPOTHESIS_FAMILY_LOCK.json')['families']['A']
    ids=plan['candidate_ids'];representations=['innovation','magnitude','sign']
    execution=['scripts/evaluate_next4_base.py','src/next4/cache.py','src/next4/frozen.py','src/next4/model.py',
        'src/next2/blend.py','src/next/data.py','src/next/guard.py','src/scoring/ts_auc.py']
    menu={'status':'FROZEN_BEFORE_FIRST_FAMILY_A_SCORE','created_utc':utc_now(),'family':'A','candidate_ids':ids,
        'representations':representations,'historical_AR_order':8,'normalization':'Existing immutable ARCH1 clipped innovation',
        'lags':[1,2,4,8],'EWMA_halflives':[8,32],'feature_count_each':18,
        'correlation':'Centered local EWMA cross covariance divided by the product of local standard deviations',
        'H_reference':'Same correlation moments in clean historical innovations; lower/upper/squared transforms fixed per recipe',
        'departure_scale':'H cross-product SD divided by H current/lag SD product, times finite-prefix EWMA scale; no calibrated p-value claim',
        'noise_floor':.1,'centered_amplitude_clip':8.,'normalized_innovation_clip':4.,'departure_clip':128.,
        'predictive_combination':'0.9 * existing BL80 float32 score + 0.1 * new expert, computed in float64 and rounded float32',
        'standalone_used_for_promotion':False,'learner':'Existing deterministic LightGBM 100 trees, 15 leaves, depth 4, W3',
        'screen_gate':{'mean_delta_min':policy['screen']['mean_delta_min'],'each_fold_delta_min':policy['screen']['each_fold_delta_min']},
        'stop_rule':'Exactly these three recipes; failed screen cannot reach full; no lag/half-life/weight/seed variations after scores.',
        'novelty':'A1 uses aggregate normalized local correlation departures, not C205 raw lag products. A2 uses local centered magnitude correlation rather than old squared-product channels. A3 uses aggregate H-relative sign-correlation energy, not I1/C211 per-lag sign products.',
        'prior_dedup_sources':{n:digest(ROOT/n) for n in ['src/next2/complement_bank.py','src/next2/dependence_runtime.py',
            'src/next3/rank_dependence.py','artifacts/next3/final_tables/prepared_v2/C205_ABLATION_RESULTS.csv',
            'artifacts/next3/final_tables/prepared_v2/MODEL_RESULTS_NEXT3.csv']},
        'feature_sources':{representation:implementation_hash({'version':'joint_corr_departure_v1','representation':representation}) for representation in representations},
        'feature_test_xml_sha256':digest(BASE/'maintenance/FAMILY_A_FEATURE_TESTS_CORRECTED.xml'),
        'feature_test_scope':'10 pytest cases include 100 randomized future-suffix/prefix/streaming cases for each of three representations; independent moment oracle; finite fixed-state degenerate streams.',
        'implementation_test_failure_preserved':'artifacts/next4/maintenance/A_oracle_001/FAILURE.json',
        'new_scores_before_registration':0,'new_scientific_fits_before_registration':0,
        'policy_sha256':digest(policy_path)}
    menu_path=BASE/'families/A_MENU_LOCK.json';atomic_json(menu_path,menu,immutable=True)
    for candidate,representation in zip(ids,representations):
        settings={'version':'joint_corr_departure_v1','representation':representation}
        hyp={'experiment_id':candidate,'parent':'BL2_C205_DEPENDENCE_80','family':'A',
            'hypothesis':'Joint H-relative '+representation+' correlation departures may provide incremental signal beyond J102/C205.',
            'causal_mechanism':'H-only AR8/ARCH normalization and lag references; each scalar update maintains bounded local moments and joint energies.',
            'exact_changes':{'feature_bank':'Joint local correlation departure','representation':representation,'existing_scientific_components_unchanged':True},
            'parameters':{'learner':'M4_lightgbm','learner_params':{'n_estimators':100,'num_leaves':15,'max_depth':4,'min_child_samples':30,
                'reg_lambda':10,'learning_rate':.05,'n_jobs':4,'random_state':20260908,'deterministic':True,'force_col_wise':True,'verbosity':-1},
                'weighting':'W3','frozen_BL80_weight':.9},
            'candidate_menu_id':'A_MENU_V1','expected_gain':'Positive fixed-blend incremental TS-AUC; admission requires frozen overall performance or robustness gates.',
            'risk':'No new raw observations; weak/no independent information, noisy short local variance, heavy tails and adaptive DEV selection may defeat the representation.',
            'screen_gate':menu['screen_gate'],'full_gate':policy['performance'],'stop_rule':menu['stop_rule'],
            'created_before_score':True,'created_utc':utc_now(),'feature_module':'src.next4.dependence','feature_settings':settings,
            'feature_names':feature_names(settings),'feature_sha256':implementation_hash(settings),'model_source_sha256':source_hash(),
            'execution_source_sha256':{n:digest(ROOT/n) for n in execution},'family_menu_sha256':digest(menu_path),
            'policy_sha256':digest(policy_path),'previous_failure_distinction':menu['novelty']}
        atomic_json(BASE/'registrations'/(candidate+'.json'),hyp,immutable=True)
    for index,candidate in enumerate(ids):
        register_job({'id':f'A{index+1:03d}_SCREEN','kind':'research','candidate_id':candidate,'family':'A',
            'argv':['scripts/evaluate_next4_base.py','--candidate',candidate,'--scope','screen'],
            'description':'Frozen Family A '+candidate+' locked screen; no automatic full after screen failure','retry_safe':False})
    progress('Family A registered and queued','Three structurally distinct dependence representations, one fixed 90/10 combination each; no new model scores before immutable registration')
    print('NEXT4 FAMILY A FROZEN',len(ids),'candidates; screen jobs queued',flush=True)


if __name__=='__main__':main()
