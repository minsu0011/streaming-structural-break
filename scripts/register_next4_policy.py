"""Freeze NEXT4 decisions and finite structural family slots before new scores."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,atomic_json,digest,read_json,utc_now,progress


def main():
    assert read_json(BASE/'preflight/NEXT4_PREFLIGHT.json')['status']=='PASS'
    reference=read_json(BASE/'preflight/FROZEN_REFERENCES.json');old=read_json(ROOT/'configs/next3/RESEARCH_POLICY_LOCK.json')
    original=reference['roles']['FROZEN_PERFORMANCE_BL80']
    policy={'version':'NEXT4_RESEARCH_POLICY_V1','created_utc':utc_now(),
        'reference_sha256':digest(BASE/'preflight/FROZEN_REFERENCES.json'),
        'user_spec_sha256':digest(ROOT/'docs/USER_SPECIFICATION_NEXT4.md'),
        'primary_metric':'official_TS_AUC','primary_reference':'BL2_C205_DEPENDENCE_80',
        'reference_mean':original['full']['mean'],'reference_folds':original['full']['fold_scores'],
        'screen':{**old['screen'],'mean_delta_min':-.001,'each_fold_delta_min':-.004,
            'reference':'Frozen primary/complement screen OOF at the identical locked 3000-series coordinates',
            'new_candidate_rule':'Only registered fixed prediction combination; standalone is diagnostic',
            'full_after_failure':False},
        'performance':{'mean_delta_min':.0013,'median_delta_min':-.0005,'worst_delta_min':-.001,
            'bootstrap_probability_min':.90,'ci_lower_strictly_positive_required':True,
            'ALT_positive_min':2,'ALT_count':3,'minimum_matched_H_regime_delta':-.01,
            'minimum_supported_major_mechanism_delta':-.005,'support_break_series_min':100,
            'leakage_pass_required':True,'prefix_pass_required':True,'deterministic_required':True,
            'maximum_direct_runtime_ratio':3.,
            'synthetic_outlier_scope':'Mandatory diagnostic and explicit warning; not an automatic veto of large real-DEV gains'},
        'generation':{'mean_delta_min':.0025,'strong_mean_delta_min':.0035,'bootstrap_probability_strict_min':.95,
            'ALT_positive_required':3,'worst_delta_min':0.,'new_independent_signal_or_clear_robustness_required':True,
            'all_performance_checks_required':True},
        'robustness':{'overall_mean_delta_min':-.0005,'SCALE_delta_min':.002,
            'SCALE_ci_lower_strict_positive':True,'SCALE_break_support_min':100,
            'minimum_supported_variance_increase_delta':-.005,'ARCH_decrease_each_delta_strict_min':0.,
            'tail_decrease_delta_strict_min':0.,'minimum_other_supported_mechanism_delta':-.005,
            'no_synthetic_only_promotion':True},
        'bootstrap':{'draws':2000,'seed':20260908,'unit':'whole_series_paired_within_existing_validation_folds',
            'scope':'Conditional fitted OOF, no adaptive-selection or model-fit uncertainty coverage',
            'generation_optional_draws':5000,'family_max_T':'exploratory conditional diagnostic, no unconditional FWER claim'},
        'nested':{'all_meta_rows_from_inner_OOF':True,'outer_validation_used_in_meta_fit':False,
            'reuse_exact_NEXT3_nested_base_fits_when_coordinates_and_source_match':True},
        'prefix_cases_per_feature_min':100,'finalist_prefix_cases_min':300,
        'model_seed':20260908,'fixed_primary_folds':[0,1,2,3,4],
        'alternate_partitions':'Existing ALT_1/ALT_2/ALT_3 and H-only regime lock, unchanged',
        'saturation':{'structural_hypotheses_min':10,'best_gain_strict_below':.0007,
            'two_warning_conditions_for_HIGH':True,'score_mining_stop_when_case_G':True},
        'research_productive_min_seconds':21600,'final_audit_min_seconds':2700,'final_audit_target_seconds':3000,
        'forbidden_inputs':['dataset_id','tau','final_online_length','future_suffix','posthoc_mechanism'],
        'no_fine_weight_search':True,'no_seed_shopping':True,'no_row_random_CV':True,
        'all_four_seals_unopened':True,'reduced_new_usage':0,'public_submission_new':0,'cloud_submission_new':0}
    families={
        'A':{'name':'Dependence/whiteness','candidate_ids':['A1_INNOVATION_PORTMANTEAU','A2_MAGNITUDE_PORTMANTEAU','A3_SIGN_PORTMANTEAU'],
            'principle':'H-centered aggregate short/long lag-correlation departures; nonlinear joint statistics rather than raw lag subset changes'},
        'B':{'name':'Robust direction asymmetry','candidate_ids':['B1_SPLIT_ENERGY_BRANCHES','B2_LOG_ENERGY_SEMITAIL'],
            'principle':'Keep lower/upper branches separately and preserve direction; no repeat of NEXT3 piecewise mapping followed by global recentering'},
        'C':{'name':'Distribution change','candidate_ids':['C1_FIXED_BIN_HELLINGER','C2_FIXED_BIN_JS','C3_SHAPE_RATIO_DISTRIBUTION'],
            'principle':'Fixed residual-space cells and separately retained H masses/divergences, distinct from original rank-uniform chi-square/CvM CDF bank'},
        'D':{'name':'Sequential likelihood','candidate_ids':['D1_DISCRETE_MEAN_MIXTURE','D2_DISCRETE_SCALE_MIXTURE','D3_HAZARD_MIXTURE'],
            'principle':'Bounded discrete likelihood alternatives or hazard recurrence; no nearby age-grid search'},
        'E':{'name':'Metric-aligned nested combination','candidate_ids':['E1_NESTED_PAIRWISE_LINEAR','E2_NESTED_PAIRWISE_INTERACTIONS'],
            'principle':'Same-time train-only pair differences of honest inner base OOF, distinct from old boosted rank learners and NEXT3 binary-loss gates'},
        'F':{'name':'Finite multi-expert combinations','candidate_ids':['F1_FIXED_INDEPENDENT_THIRD','F2_FIXED_JOINT_NEW_EXPERTS'],
            'principle':'Conditional on registered independent expert evidence; frozen small combinations, no dynamic gate or percent-grid'},
        'G':{'name':'H-only context','candidate_ids':['G1_H_CONTEXT_BASE','G2_H_CONTEXT_NESTED_PAIRWISE'],
            'principle':'Small immutable H summaries; distinguish direct base covariates from nested learned forecast combination'},
        'H':{'name':'Compact temporal evidence','candidate_ids':['H1_FIXED_LINEAR_STATE_READOUT'],
            'principle':'Small causal state recurrence and train-fold readout; no GRU/LSTM/Transformer'}
    }
    family_lock={'status':'FROZEN_STRUCTURAL_FAMILY_PLAN_BEFORE_NEXT4_SCORES','created_utc':utc_now(),
        'families':families,'family_count':8,'named_candidate_slots':sum(len(x['candidate_ids']) for x in families.values()),
        'family_contract':'Each family exact numeric menu/source/stop rule must be frozen before its first score; named slots may be rejected as duplicates or conditionally skipped, never silently replaced after scoring.',
        'unused_candidates_do_not_count_as_completed_hypotheses':True,'maximum_active_bottlenecks':2,
        'policy_sha256':None}
    target=ROOT/'configs/next4/RESEARCH_POLICY_LOCK.json';atomic_json(target,policy,immutable=True)
    family_lock['policy_sha256']=digest(target)
    for p in (ROOT/'configs/next4/HYPOTHESIS_FAMILY_LOCK.json',ROOT/'HYPOTHESIS_FAMILY_LOCK.json'):
        atomic_json(p,family_lock,immutable=True)
    budget={'status':'FROZEN_BEFORE_NEXT4_SCORES','created_utc':utc_now(),'major_families_max':8,
        'new_full_candidates_max':24,'named_slots':family_lock['named_candidate_slots'],
        'per_family_default_max':4,'candidate_identity':'One frozen model/feature/learner/prediction-combination recipe; every separately selected blend or learner is a candidate.',
        'standalone_diagnostic_not_separately_selectable':True,
        'fit_count_separate':'Every attempted/completed training operation is recorded; cache reuse is zero new fits.',
        'family_lock_sha256':digest(ROOT/'configs/next4/HYPOTHESIS_FAMILY_LOCK.json')}
    for p in (ROOT/'configs/next4/CANDIDATE_BUDGET.json',ROOT/'CANDIDATE_BUDGET.json'):atomic_json(p,budget,immutable=True)
    progress('NEXT4 scientific menu and gates frozen','8 structural families, 18 named recipes within 24 full-candidate budget; performance +0.0013 and generation +0.0025 unchanged after scores')
    print('NEXT4 POLICY FROZEN',family_lock['named_candidate_slots'],'named slots; no new scores',flush=True)


if __name__=='__main__':main()
