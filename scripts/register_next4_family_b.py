"""Prospective finite directional-energy menu, with a J102 replacement learner."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,atomic_json,digest,read_json,utc_now,progress
from src.next4.scale_asymmetry import implementation_hash,feature_names
from src.next4.model import source_hash
from src.next4.scheduler import register_job


def main():
    policy_path=ROOT/'configs/next4/RESEARCH_POLICY_LOCK.json';policy=read_json(policy_path)
    ids=read_json(ROOT/'configs/next4/HYPOTHESIS_FAMILY_LOCK.json')['families']['B']['candidate_ids']
    kinds=['split_energy','log_semitail']
    xml=BASE/'maintenance/FAMILY_B_FEATURE_TESTS.xml'
    suites=list(ET.parse(xml).getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==7
    assert all(int(s.get('failures',0))+int(s.get('errors',0))==0 for s in suites)
    execution=['scripts/evaluate_next4_replacement.py','src/next4/cache.py','src/next4/frozen.py','src/next4/model.py',
        'src/next2/blend.py','src/next/data.py','src/next/guard.py','src/scoring/ts_auc.py']
    params={'learner':'M4_lightgbm','learner_params':{'n_estimators':100,'num_leaves':15,'max_depth':4,'min_child_samples':30,
        'reg_lambda':10,'learning_rate':.05,'n_jobs':4,'random_state':20260908,'deterministic':True,'force_col_wise':True,'verbosity':-1},
        'weighting':'W3','new_J102_weight':.8}
    menu={'status':'FROZEN_BEFORE_FIRST_FAMILY_B_SCORE','created_utc':utc_now(),'family':'B','candidate_ids':ids,
        'kinds':kinds,'historical_AR_order':8,'ARCH_order':1,'unchanged_J102_features':77,'new_features':16,
        'channels':['raw_energy_ratio','ARCH_standardized_energy_ratio'],'H_center':'median of raw or log(ratio+0.0001) channel',
        'directional_scale':'Median strictly positive upper/lower deviation separately; minimum 0.0001',
        'branches':'Positive upper and lower amplitudes retained separately; clipped at 12 before H branch reference',
        'online_features_each_branch':['current nonnegative amplitude','H-relative EWMA8','H-relative EWMA32','log1p one-sided CUSUM drift 0.25'],
        'H_branch_reference':'H branch mean and median absolute deviation from that mean, scale minimum 0.1',
        'predictive_combination':'0.8 * newly fitted augmented J102 expert + 0.2 * frozen C205 OOF; float64 sum rounded float32',
        'standalone_used_for_promotion':False,'learner_parameters':params,
        'screen_gate':{k:policy['screen'][k] for k in ['mean_delta_min','each_fold_delta_min']},
        'stop_rule':'Exactly two energy representations. Failed screen stops. No branch, center, half-life, tree, seed or blend-weight variants after scores.',
        'novelty':'NEXT3 S1 joined separately scaled directions and globally recentered them, losing a pure directional channel. These recipes retain positive upper/lower branches as four separate processes. B2 additionally uses a fixed log-energy map, not S2 lower-quartile occupancy.',
        'prior_dedup_sources':{n:digest(ROOT/n) for n in ['src/next3/scale_calibration.py','src/next3/scale_direction.py',
            'artifacts/next3/final_tables/prepared_v2/SCALE_REGIME_BREAKDOWN_NEXT3.csv'] if (ROOT/n).exists()},
        'feature_sources':{kind:implementation_hash({'version':'separate_energy_branches_v1','kind':kind}) for kind in kinds},
        'feature_test_xml_sha256':digest(xml),'feature_test_scope':'Seven passing cases; 100 randomized prefix cases for each recipe, independent first-77-feature parity, finite degenerate streams, directional branch oracle.',
        'new_scores_before_registration':0,'new_scientific_fits_before_registration':0,'policy_sha256':digest(policy_path)}
    menu_path=BASE/'families/B_MENU_LOCK.json';atomic_json(menu_path,menu,immutable=True)
    for candidate,kind in zip(ids,kinds):
        settings={'version':'separate_energy_branches_v1','kind':kind}
        hyp={'experiment_id':candidate,'parent':'BL2_C205_DEPENDENCE_80','family':'B',
            'hypothesis':'Separate downward and upward energy evidence can recover SCALE_DECREASE sensitivity without suppressing variance increases.',
            'causal_mechanism':'Immutable H-derived AR/ARCH and positive directional branches; scalar online updates; all learner labels from outer training groups only.',
            'exact_changes':{'feature_bank':'J10277 plus 16 separate energy branch features','kind':kind,'combination':'0.8 new J102 extension + 0.2 frozen C205'},
            'parameters':params,'candidate_menu_id':'B_MENU_V1',
            'expected_gain':'Fixed-blend TS-AUC performance improvement or the preregistered supported scale-decrease robustness role.',
            'risk':'Low-energy ratios are noisy; re-fitting J102 may disrupt its ranking. H-direction asymmetry can distort ARCH and tail mechanisms.',
            'screen_gate':menu['screen_gate'],'full_gate':policy['performance'],'robustness_gate':policy['robustness'],
            'stop_rule':menu['stop_rule'],'created_before_score':True,'created_utc':utc_now(),
            'feature_module':'src.next4.scale_asymmetry','feature_settings':settings,'feature_names':feature_names(settings),
            'feature_sha256':implementation_hash(settings),'model_source_sha256':source_hash(),
            'execution_source_sha256':{n:digest(ROOT/n) for n in execution},'family_menu_sha256':digest(menu_path),
            'policy_sha256':digest(policy_path),'previous_failure_distinction':menu['novelty']}
        atomic_json(BASE/'registrations'/(candidate+'.json'),hyp,immutable=True)
    for index,candidate in enumerate(ids):
        register_job({'id':f'B{index+1:03d}_SCREEN','kind':'research','candidate_id':candidate,'family':'B',
            'argv':['scripts/evaluate_next4_replacement.py','--candidate',candidate,'--scope','screen'],
            'description':'Registered separate directional energy features with unchanged 80/20 combination','retry_safe':False})
    progress('Family B frozen and queued','Two candidates, no scores before registration; lower/upper evidence retained as separate features')
    print('NEXT4 FAMILY B FROZEN',len(ids),'screen jobs queued',flush=True)


if __name__=='__main__':main()
