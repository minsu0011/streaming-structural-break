"""Two frozen H-context uses, one direct and one genuinely nested."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,atomic_json,read_json,digest,utc_now,progress
from src.next4.historical_context import implementation_hash,feature_names,CONTEXT_NAMES
from src.next4.model import source_hash as base_source_hash
from src.next4.context_pairwise import source_hash as meta_source_hash,SETTINGS as META_SETTINGS
from src.next4.nested_reuse import source_hash as nested_source_hash
from src.next4.scheduler import register_job


def checked_xml(path,count):
    suites=list(ET.parse(path).getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==count and all(int(s.get('failures',0))+int(s.get('errors',0))==0 for s in suites)
    return digest(path)


def main():
    policy_path=ROOT/'configs/next4/RESEARCH_POLICY_LOCK.json';policy=read_json(policy_path)
    ids=read_json(ROOT/'configs/next4/HYPOTHESIS_FAMILY_LOCK.json')['families']['G']['candidate_ids']
    context_settings={'version':'historical_context_v1','kind':'base'}
    base_params={'learner':'M4_lightgbm','learner_params':{'n_estimators':100,'num_leaves':15,'max_depth':4,'min_child_samples':30,
        'reg_lambda':10,'learning_rate':.05,'n_jobs':4,'random_state':20260908,'deterministic':True,'force_col_wise':True,'verbosity':-1},
        'weighting':'W3','new_J102_weight':.8}
    runners={ids[0]:'scripts/evaluate_next4_replacement.py',ids[1]:'scripts/evaluate_next4_context_pairwise.py'}
    menu={'status':'FROZEN_BEFORE_FIRST_FAMILY_G_REAL_SCORE','created_utc':utc_now(),'family':'G','candidate_ids':ids,
        'context_features':list(CONTEXT_NAMES),'context_count':10,'context_source_sha256':implementation_hash(context_settings),
        'context_construction':'Each clean H alone: log std, abs-residual q95/q50, centered acf1/acf4, fitted AR8 coefficients 1/2, ARCH1 alpha, Fourier low/high energy, log H length, ARCH-normalized absolute-tail rate >2',
        'context_scaling':'Fixed clips/divisors in source, no cross-series fitted normalization, cluster, IDs, online values or outcome labels',
        'spectral_cutoff_cycles_per_point':.125,'H_length_used':True,'online_length_used':False,
        'base_recipe':'Original J10277 plus all ten immutable H summaries, same W3 LightGBM recipe; 0.8 newly fitted extended J + 0.2 frozen C205',
        'meta_recipe':'Original J/C scores plus ten features (J-C)*H_context. Twelve coefficients, same-time pairwise objective on honest inner OOF only.',
        'meta_context_main_effects':False,'meta_pair_settings':META_SETTINGS,'meta_ridge_anchor':[.8,.2]+[0.]*10,
        'meta_bounds':[[0,4],[0,4]]+[[-1,1]]*10,'meta_output':'sigmoid(f-0.5), float32',
        'base_parameters':base_params,'screen_gate':{k:policy['screen'][k] for k in ['mean_delta_min','each_fold_delta_min']},
        'stop_rule':'Exactly one base-context extension and one H-modulated pairwise map. No context subset, cluster, cutoff, ridge, loss, weight or seed search after scores; failed screen stops.',
        'novelty':'The baseline uses H to normalize each stream but lacks these explicit small H covariates. G2 modulates only the relative expert score gap with outcome-blind H; it differs from E2 current-score polynomial terms and the old outcome-learned dynamic gates.',
        'prior_dedup_sources':{p:digest(ROOT/p) for p in ['configs/historical_context.json','src/next3/router.py','src/next3/learned_router.py','src/next4/pairwise.py']},
        'base_test_xml_sha256':checked_xml(BASE/'maintenance/FAMILY_G_CONTEXT_TESTS_CORRECTED.xml',7),
        'meta_test_xml_sha256':checked_xml(BASE/'maintenance/FAMILY_G_META_TESTS_RUNTIME_V2.xml',5),
        'test_scope':'100 raw-history base-prefix/streaming/J10277 cases, 100 H-modulated current-score prefix cases, H translation/formula checks, context coordinate shuffle rejection and a synthetic optimizer/serialization test',
        'preserved_pre_score_issue':'artifacts/next4/maintenance/G_arch_adapter_001/FAILURE.json',
        'scalar_runtime_pre_score_revision':'artifacts/next4/maintenance/G_META_RUNTIME_V2/CHANGE.json',
        'synthetic_meta_unit_fits_before_registration':2,'real_G_model_fits_before_registration':0,'real_G_quality_scores_before_registration':0,
        'evaluation_runners':runners,'dispatch_priority':50,'policy_sha256':digest(policy_path)}
    menu_path=BASE/'families/G_MENU_LOCK.json';atomic_json(menu_path,menu,immutable=True)
    for index,candidate in enumerate(ids):
        meta=index==1
        execution=[runners[candidate],'src/next4/frozen.py','src/next/data.py','src/next/guard.py','src/scoring/ts_auc.py']
        if not meta:execution+=['src/next4/cache.py','src/next4/model.py','src/next2/blend.py']
        hyp={'experiment_id':candidate,'parent':'BL2_C205_DEPENDENCE_80','family':'G',
            'hypothesis':'Outcome-blind H context can improve '+('relative expert allocation' if meta else 'base ranking')+' across heterogeneous historical regimes.',
            'causal_mechanism':'Immutable summaries of clean H; '+('only honest inner OOF component scores and training labels fit the pairwise coefficients' if meta else 'only outer-training labels fit the augmented base tree'),
            'exact_changes':{'context_names':list(CONTEXT_NAMES),'use':'nested gap modulation' if meta else 'base feature extension'},
            'parameters':dict(META_SETTINGS) if meta else base_params,'candidate_menu_id':'G_MENU_V1',
            'expected_gain':'Stable primary, ALT and held-out H-regime TS-AUC; context may represent differences hidden by per-series normalization.',
            'risk':'Historical summaries may proxy accidental cohort structure; held-out H regimes can expose poor extrapolation; inner/outer base training sizes differ for meta models.',
            'screen_gate':menu['screen_gate'],'full_gate':policy['performance'],'stop_rule':menu['stop_rule'],
            'created_before_score':True,'created_utc':utc_now(),'model_source_sha256':meta_source_hash() if meta else base_source_hash(),
            'execution_source_sha256':{p:digest(ROOT/p) for p in execution},'family_menu_sha256':digest(menu_path),
            'policy_sha256':digest(policy_path),'previous_failure_distinction':menu['novelty']}
        if meta:hyp['nested_source_sha256']=nested_source_hash()
        else:hyp.update(feature_module='src.next4.historical_context',feature_settings=context_settings,
            feature_names=feature_names(context_settings),feature_sha256=implementation_hash(context_settings))
        atomic_json(BASE/'registrations'/(candidate+'.json'),hyp,immutable=True)
    for index,candidate in enumerate(ids):
        job={'id':f'G{index+1:03d}_SCREEN','kind':'research','family':'G','candidate_id':candidate,'priority':50,
            'argv':[runners[candidate],'--candidate',candidate,'--scope','screen'],
            'description':'Frozen H-only base or honest nested score-gap context candidate','retry_safe':False}
        if index==1:job['depends_on_success']=['E000_NESTED_REUSE_AUDIT']
        register_job(job)
    progress('Family G registered','Two fixed H-only recipes; no target-derived clusters, no online length, nested G2 only after audited 56-fit reuse')
    print('NEXT4 FAMILY G FROZEN',len(ids),'screen jobs queued',flush=True)


if __name__=='__main__':main()
