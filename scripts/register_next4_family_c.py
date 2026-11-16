"""Freeze a finite distribution family before its first model score."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,atomic_json,digest,read_json,utc_now,progress
from src.next4.distribution import implementation_hash,feature_names
from src.next4.model import source_hash
from src.next4.scheduler import register_job


def main():
    policy_path=ROOT/'configs/next4/RESEARCH_POLICY_LOCK.json';policy=read_json(policy_path)
    ids=read_json(ROOT/'configs/next4/HYPOTHESIS_FAMILY_LOCK.json')['families']['C']['candidate_ids']
    kinds=['hellinger','js','shape_ratio'];xml=BASE/'maintenance/FAMILY_C_FEATURE_TESTS.xml'
    suites=list(ET.parse(xml).getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==10 and all(int(s.get('failures',0))+int(s.get('errors',0))==0 for s in suites)
    execution=['scripts/evaluate_next4_base.py','src/next4/cache.py','src/next4/frozen.py','src/next4/model.py',
        'src/next2/blend.py','src/next/data.py','src/next/guard.py','src/scoring/ts_auc.py']
    params={'learner':'M4_lightgbm','learner_params':{'n_estimators':100,'num_leaves':15,'max_depth':4,'min_child_samples':30,
        'reg_lambda':10,'learning_rate':.05,'n_jobs':4,'random_state':20260908,'deterministic':True,'force_col_wise':True,'verbosity':-1},
        'weighting':'W3','frozen_BL80_weight':.9}
    menu={'status':'FROZEN_BEFORE_FIRST_FAMILY_C_SCORE','created_utc':utc_now(),'family':'C','candidate_ids':ids,'kinds':kinds,
        'historical_AR_order':8,'normalization':'Immutable H ARCH1 clipped normalized innovations',
        'edges':[-2.,-1.,-.5,0.,.5,1.,2.],'cells':8,'assignment':'searchsorted right',
        'H_reference':'Empirical fixed-cell masses with 0.5 pseudocount in every cell, normalized once and immutable',
        'EWMA_halflives':[8,32,128],'initial_online_mass':'H reference, no future-prefix initialization',
        'distance_definitions':{'hellinger':'Squared Hellinger, 0.5 sum((sqrt(p)-sqrt(q))**2)',
            'js':'Jensen-Shannon divergence with natural logarithms; not its square-root distance',
            'shape_ratio':'H-relative center/moderate and extreme/moderate log ratios, directional tail imbalance, masses and total variation'},
        'center_cells':[3,4],'moderate_cells':[1,2,5,6],'extreme_cells':[0,7],
        'feature_count':26,'distance_features':'Eight summaries per half-life and current H cell rarity plus signed cell index',
        'predictive_combination':'0.9 * frozen BL80 + 0.1 * new distribution expert, float64 sum rounded float32',
        'standalone_used_for_promotion':False,'learner_parameters':params,
        'screen_gate':{k:policy['screen'][k] for k in ['mean_delta_min','each_fold_delta_min']},
        'stop_rule':'Exactly these three recipes; no alternate bin counts, edges, smoothing, half-lives or blend weights after scores; failed screen stops.',
        'novelty':'Old complement CDF bank used empirical H midranks with approximately uniform H bins and chi-square/CvM. These bins are fixed in standardized residual space, preserve nonuniform H masses, and use square-root/logarithmic distances or center/moderate/extreme shape ratios.',
        'prior_dedup_sources':{'src/next2/complement_bank.py':digest(ROOT/'src/next2/complement_bank.py')},
        'methods_source':'https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html',
        'feature_test_xml_sha256':digest(xml),'feature_test_scope':'10 tests including 100 random prefix cases per recipe, independent SciPy JS and NumPy Hellinger/mass-recurrence oracles, immutable H reference and bounded degenerate state.',
        'feature_sources':{kind:implementation_hash({'version':'fixed_residual_cells_v1','kind':kind}) for kind in kinds},
        'new_scores_before_registration':0,'new_scientific_fits_before_registration':0,'policy_sha256':digest(policy_path)}
    menu_path=BASE/'families/C_MENU_LOCK.json';atomic_json(menu_path,menu,immutable=True)
    for candidate,kind in zip(ids,kinds):
        settings={'version':'fixed_residual_cells_v1','kind':kind}
        hyp={'experiment_id':candidate,'parent':'BL2_C205_DEPENDENCE_80','family':'C',
            'hypothesis':'Fixed residual-space distribution '+kind+' can detect shape changes beyond energy and serial dependence.',
            'causal_mechanism':'H alone fixes bin masses and normalization; each online scalar updates three bounded mass vectors.',
            'exact_changes':{'feature_bank':'26 fixed residual-bin distribution features','kind':kind},'parameters':params,
            'candidate_menu_id':'C_MENU_V1','expected_gain':'Positive incremental TS-AUC in fixed 90/10 combination, with tail/skew subgroup evidence.',
            'risk':'Much histogram change may duplicate scale features; sparse tail cells and marginal distribution shifts can be weak complements.',
            'screen_gate':menu['screen_gate'],'full_gate':policy['performance'],'stop_rule':menu['stop_rule'],
            'created_before_score':True,'created_utc':utc_now(),'feature_module':'src.next4.distribution','feature_settings':settings,
            'feature_names':feature_names(settings),'feature_sha256':implementation_hash(settings),'model_source_sha256':source_hash(),
            'execution_source_sha256':{n:digest(ROOT/n) for n in execution},'family_menu_sha256':digest(menu_path),
            'policy_sha256':digest(policy_path),'previous_failure_distinction':menu['novelty']}
        atomic_json(BASE/'registrations'/(candidate+'.json'),hyp,immutable=True)
    for index,candidate in enumerate(ids):
        register_job({'id':f'C{index+1:03d}_SCREEN','kind':'research','candidate_id':candidate,'family':'C',
            'argv':['scripts/evaluate_next4_base.py','--candidate',candidate,'--scope','screen'],
            'description':'Registered fixed-bin distribution expert with frozen 90/10 combination','retry_safe':False})
    progress('Family C registered','Three finite distribution recipes queued; independent formula and prefix tests passed before scoring')
    print('NEXT4 FAMILY C FROZEN',len(ids),'screen jobs queued',flush=True)


if __name__=='__main__':main()
