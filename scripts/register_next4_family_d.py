"""Prospective finite likelihood alternatives; no post-score age-grid changes."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,atomic_json,digest,read_json,utc_now,progress
from src.next4.sequential import implementation_hash,feature_names
from src.next4.model import source_hash
from src.next4.scheduler import register_job


def main():
    policy_path=ROOT/'configs/next4/RESEARCH_POLICY_LOCK.json';policy=read_json(policy_path)
    ids=read_json(ROOT/'configs/next4/HYPOTHESIS_FAMILY_LOCK.json')['families']['D']['candidate_ids']
    kinds=['mean_mixture','scale_mixture','hazard_mixture'];xml=BASE/'maintenance/FAMILY_D_FEATURE_TESTS_CORRECTED.xml'
    suites=list(ET.parse(xml).getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==11 and all(int(s.get('failures',0))+int(s.get('errors',0))==0 for s in suites)
    execution=['scripts/evaluate_next4_base.py','src/next4/cache.py','src/next4/frozen.py','src/next4/model.py',
        'src/next2/blend.py','src/next/data.py','src/next/guard.py','src/scoring/ts_auc.py']
    params={'learner':'M4_lightgbm','learner_params':{'n_estimators':100,'num_leaves':15,'max_depth':4,'min_child_samples':30,
        'reg_lambda':10,'learning_rate':.05,'n_jobs':4,'random_state':20260908,'deterministic':True,'force_col_wise':True,'verbosity':-1},
        'weighting':'W3','frozen_BL80_weight':.9}
    menu={'status':'FROZEN_BEFORE_FIRST_FAMILY_D_SCORE','created_utc':utc_now(),'family':'D','candidate_ids':ids,'kinds':kinds,
        'historical_AR_order':8,'normalization':'Immutable H ARCH1 normalized innovations, then H mean/std and clip at +/-6',
        'sparse_ages':[4,8,16,32,64,128,256,512],'early_age':'min(online points seen, fixed age); duplicated early windows retained',
        'mean_amplitudes':[-1.5,-.75,-.25,.25,.75,1.5],'variance_multipliers':[.5,.7,1.5,2.],
        'mean_log_likelihood':'mu * sum(z) - 0.5 * n * mu**2',
        'scale_log_likelihood':'-0.5 * n * log(v) + 0.5 * (1-1/v) * sum(z**2)',
        'mixture_weights':'Uniform alternatives within a fixed age; uniform sparse ages for all-age summaries',
        'hazards':[1./32,1./128,1./512],'hazard_alternatives':'Six pure mean and four pure scale alternatives, each with probability 0.1 on first change',
        'hazard_recurrence':'Three parallel single-change absorbing filters in log space, each with a fixed geometric hazard; no post-change resets',
        'hazard_readout':'Five null-relative log odds and conditional-alternative entropy per hazard; two short-long contrasts. A fixed tree learner combines the three filters.',
        'posterior_claim':'Working-model ranking features only; not calibrated real-world change probabilities, not the full Adams/MacKay run-length algorithm',
        'feature_counts':[36,36,20],'state':'Two cumulative-prefix rings of 513 float64 values, fixed counters, and three 11-state hazard vectors',
        'cost':'O(8*6) or O(8*4) or O(3*10) per point; no sequence-length loop',
        'predictive_combination':'0.9 * frozen BL80 + 0.1 * new likelihood expert; float64 sum rounded float32',
        'standalone_used_for_promotion':False,'learner_parameters':params,
        'screen_gate':{k:policy['screen'][k] for k in ['mean_delta_min','each_fold_delta_min']},
        'stop_rule':'Three fixed likelihood models only; failed screen stops; no alternate amplitudes, scales, hazards, ages, priors or blend weights after scores.',
        'novelty':'D1 replaces continuous Gaussian location-prior marginalization with separated fixed amplitudes. D2 uses a discrete variance likelihood rather than prior directional GLR or inverse-gamma evidence. D3 recursively carries ten persistent post-change alternatives under three fixed hazards, unlike independent retrospective age summaries.',
        'prior_dedup_sources':{p:digest(ROOT/p) for p in ['src/next/channel_evidence.py','src/next/serial_channel_evidence.py','src/next2/scale_likelihood.py','src/next3/negative_bayes.py']},
        'method_context':'https://arxiv.org/abs/0710.3742; NEXT4 uses a smaller explicitly different single-change state model',
        'feature_test_xml_sha256':digest(xml),'feature_test_scope':'11 tests, 100 random prefix cases for each recipe, direct sparse-window slice/SciPy logsumexp oracle through two ring wraps, probability-space hazard oracle, 16384-point finite bounded-state cases.',
        'pre_score_failure_preserved':'artifacts/next4/maintenance/D_scalar_clip_001/FAILURE.json',
        'additional_required_diagnostics':'Fitted-score long-null lengths 1024/4096/16384 and isolated/outlier-cluster counterexamples; a finite feature test does not replace fitted-score diagnostics.',
        'feature_sources':{kind:implementation_hash({'version':'discrete_likelihood_v1','kind':kind}) for kind in kinds},
        'new_scores_before_registration':0,'new_scientific_fits_before_registration':0,'policy_sha256':digest(policy_path)}
    menu_path=BASE/'families/D_MENU_LOCK.json';atomic_json(menu_path,menu,immutable=True)
    for candidate,kind in zip(ids,kinds):
        settings={'version':'discrete_likelihood_v1','kind':kind}
        hyp={'experiment_id':candidate,'parent':'BL2_C205_DEPENDENCE_80','family':'D',
            'hypothesis':'Fixed '+kind+' evidence can represent persistent changes differently from symmetric energy and continuous-prior age evidence.',
            'causal_mechanism':'Only H fixes normalization; past/current scalar residuals update bounded sums or persistent fixed-alternative states.',
            'exact_changes':{'feature_bank':'Discrete likelihood or fixed-hazard features','kind':kind},'parameters':params,
            'candidate_menu_id':'D_MENU_V1','expected_gain':'Positive fixed-blend TS-AUC with persistence signal distinct from the frozen baseline.',
            'risk':'Gaussian likelihood misspecification, transient outlier memory, and absorbing hazard saturation can harm long-null calibration.',
            'screen_gate':menu['screen_gate'],'full_gate':policy['performance'],'stop_rule':menu['stop_rule'],
            'created_before_score':True,'created_utc':utc_now(),'feature_module':'src.next4.sequential','feature_settings':settings,
            'feature_names':feature_names(settings),'feature_sha256':implementation_hash(settings),'model_source_sha256':source_hash(),
            'execution_source_sha256':{p:digest(ROOT/p) for p in execution},'family_menu_sha256':digest(menu_path),
            'policy_sha256':digest(policy_path),'previous_failure_distinction':menu['novelty']}
        atomic_json(BASE/'registrations'/(candidate+'.json'),hyp,immutable=True)
    for index,candidate in enumerate(ids):
        register_job({'id':f'D{index+1:03d}_SCREEN','kind':'research','candidate_id':candidate,'family':'D',
            'argv':['scripts/evaluate_next4_base.py','--candidate',candidate,'--scope','screen'],
            'description':'Frozen sparse discrete-likelihood or persistent hazard feature expert','retry_safe':False})
    progress('Family D registered','Three fixed likelihood recipes queued after 11 independent/prefix tests; no age or amplitude search')
    print('NEXT4 FAMILY D FROZEN',len(ids),'screen jobs queued',flush=True)


if __name__=='__main__':main()
