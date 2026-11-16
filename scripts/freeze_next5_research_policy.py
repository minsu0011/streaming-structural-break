"""Freeze finite scientific menus, gates and fair reference protocol before scores."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next5.artifacts import BASE,atomic_json,read_json,digest,utc_now,progress


def main():
    out=ROOT/'configs/next5';out.mkdir(parents=True,exist_ok=False)
    parameters=read_json(ROOT/'artifacts/next4/registrations/B2_LOG_ENERGY_SEMITAIL.json')['parameters']['learner_params']
    families={
      'S':{'question':'Can directional Gaussian/t scale likelihood and H-relative semitail evidence unify B2/D2 information?',
           'slots':['S1_DIRECTIONAL_SCALE','S2_STUDENT_SCALE','S4A_AUGMENTED_J_SCALE','S4B_UNIFIED_SCALE_SPECIALIST'],
           'menu':{'variance_ratios':[.5,.7,1.5,2.0],'null_variance_ratio':1.,'windows':[8,32,128],
                   'fixed_t_df':5.,'channels':['raw_AR_innovation','ARCH_standardized_innovation'],
                   'S4A':'J10277 plus S2 scale features; replace J at fixed 0.8 with C205 at 0.2',
                   'S4B':'Joint Gaussian/t directional features plus B2 semitail descriptors; fixed 0.1 specialist with ResearchPool BL80',
                   'S3':'Within-time B2/D2 feature/score and conditional pair-order decomposition only; no extra learner or prediction blending search'}},
      'P':{'question':'Can confirmation distinguish persistent changes from isolated and repeated shocks?',
           'slots':['P1_TWO_TIMESCALE_CONFIRMATION','P2_SOFT_PERSISTENCE_COUNT','P3_SHOCK_RECOVERY'],
           'menu':{'fast_slow_half_lives':[8,64],'count_window':32,'threshold':'H-only fixed 0.9 quantile of directional evidence',
                   'shock_half_life':4,'persistent_half_life':64,'directions':['decrease','increase'],'no_score_dependent_reset':True}},
      'W':{'question':'Can bounded recent-window scale estimates recover onset without unbounded contamination?',
           'slots':['W1_RECENT_SCALE_RATIO','W2_TRIMMED_RECENT_ENERGY','W3_MULTIWINDOW_DISAGREEMENT'],
           'menu':{'windows':[16,32,64,128],'trim_threshold':'H-only 0.95 energy quantile','pairs':[[16,128],[32,64]],'full_sort_per_point':False}},
      'E':{'question':'Can fixed mixture evidence represent recent change without the absorbing-hazard null pressure seen in D3?',
           'slots':['E1_SCALE_MIXTURE_EVIDENCE','E2_SPARSE_START_MIXTURE'],
           'menu':{'variance_ratios':[.5,.7,1.5,2.0],'ages':[1,2,4,8,16,32,64,128,256],
                   'mixture':'fixed prior logsumexp, no maximum-based tuning','theoretical_claim':'working likelihood features; no real-DGP anytime validity asserted'}},
      'R':{'question':'Can H-only tail calibration remove heavy-tail sensitivity without context-only target shortcuts?',
           'slots':['R1_H_ADAPTIVE_TAIL','R2_CONDITIONAL_SCALE_PIT','R3_QUANTILE_SELF_NORMALIZED'],
           'menu':{'tail_df':'deterministic H moment/quantile estimate clipped to [4,30], no outcome use',
                   'PIT':'H empirical innovation-energy CDF, lower/upper surprise separately',
                   'quantile_scale':[.25,.5,.75],'null_parameters_only':True}},
      'M':{'question':'Can bounded signed evidence add a location signal distinct from D1 Gaussian-amplitude variants?',
           'slots':['M1_ROBUST_LOCATION_LIKELIHOOD','M2_SIGN_RUN_EVIDENCE'],
           'menu':{'M1':'fixed t5 location mixture with shifts [-0.75,0.75]',
                   'M2':'H-relative sign imbalance and run persistence','half_lives':[8,32,128],
                   'screen_standalone_min':.52}},
      'X':{'question':'Do independently supported scale/persistence/tail/dependence representations interact?',
           'slots':['X1_SCALE_PERSISTENCE','X2_SCALE_TAIL','X3_DEPENDENCE_SCALE'],
           'menu':{'admission':'Both prerequisite families must show ResearchPool full positive increment >=0.0003 or standalone >=0.54; freeze exact parents before X score.',
                   'parents':'Deterministic highest full mean gain among eligible family members; ties lexicographic experiment ID',
                   'interactions':'A small explicitly registered coordinate set, never all-pairs expansion'}}}
    policy={'status':'FROZEN_NEXT5_RESEARCH_POLICY','created_utc':utc_now(),'source_sha256':digest(Path(__file__)),
        'user_specification_sha256':digest(ROOT/'docs/USER_SPECIFICATION_NEXT5.md'),
        'shadow_lock_sha256':digest(ROOT/'NEXT5_SHADOW_LOCK.json'),'split_lock_sha256':digest(ROOT/'NEXT5_SPLIT_LOCK.json'),
        'randomness_lock_sha256':digest(ROOT/'NEXT5_RANDOMNESS_LOCK.json'),
        'primary_metric':'official_TS_AUC','selection_authority':'ResearchPool grouped five-fold equally weighted mean; original full-DEV scores are historical only',
        'learner':'M4_lightgbm','learner_params':parameters,'weighting':'W3',
        'default_combination':{'ResearchPool_BL80':.9,'new_specialist':.1},
        'screen':{'mean_delta_min':-.001,'each_fold_delta_min':-.004,'folds':[0,4],'screen_series':2560,
                  'screen_IDS_source':'NEXT5_SPLIT_LOCK.json','M_standalone_min':.52},
        'performance_gate':{'mean_delta_min':.0012,'median_delta_min':0.,'worst_delta_min':-.001,
            'bootstrap_probability_min':.9,'ci_lower_strict_min':-.0003,'ALT_positive_required':2,'ALT_count':2,
            'minimum_matched_H_regime_delta':-.01,'leakage_prefix_determinism_required':True,'maximum_runtime_ratio':3.},
        'strong_gate':{'mean_delta_min':.0025,'ci_lower_strict_min':0.,'positive_folds_min':4,'no_major_collapse':True},
        'catastrophic_gate':{'supported_mechanism_break_series_min':100,'supported_mechanism_delta_strict_below':-.01,
            'supported_H_regime_break_series_min':100,'supported_H_regime_delta_strict_below':-.02,
            'matched_H_exclusion_delta_strict_below':-.01,
            'synthetic_warning':'Report all losses; fixed outlier/long-null diagnostics do not silently replace primary TS-AUC.'},
        'shadow_confirmation':{'directional_delta_strict_min':0.,'strong_delta_min':.001,'rejection_delta_max':-.001,
            'no_supported_catastrophic_collapse':True,'otherwise':'INCONCLUSIVE','maximum_new_finalists':3,
            'all_roles_require_common_research_finalist_gate':True,'no_eligible_candidate':'Keep Shadow CLOSED; do not spend it to rescue a near-miss.',
            'no_tuning_after_first_Shadow_access':True,'shadow_bootstrap':'No new Shadow bootstrap; paired point estimates and predefined slice diagnostics only'},
        'finalist_roles':['PERFORMANCE','ROBUSTNESS','ORTHOGONAL'],
        'finalist_diversity':'At most one near-identical recipe from a family; select distinct scientific questions, no mandatory three slots',
        'winner_gap_uncertain_below':.0007,'bootstrap':{'replicates':2000,'seed_base':20260908,'unit':'whole series','new_draws_after_score':False},
        'saturation':{'structural_hypotheses_min':12,'best_gain_strict_below':.0007,'action':'Stop new scientific fits; continue meaningful ResearchPool robustness/reproduction/analysis.'},
        'time_bins':[[0,9],[10,24],[25,49],[50,99],[100,199],[200,399],[400,None]],
        'age_bins':[[0,4],[5,9],[10,19],[20,49],[50,None]],
        'support_break_series_min':100,'prefix_cases_per_family_min':100,'finalist_prefix_cases_min':300,
        'long_null_horizons':[1024,4096,16384],'outlier_impulse_lags':[0,1,2,4,8,16,32,64,128,256,512],
        'minimum_productive_research_seconds':36000,'separate_final_audit_seconds_min':3000,
        'preferred_shadow_timing':'After >=36000 observed research seconds and all required ResearchPool selection work, at most three frozen finalists',
        'new_fits_on_Shadow':0,'selected_model_training_scope':'ResearchPool only; fresh reproduction uses raw ResearchPool. No optional all-DEV refit in this session.',
        'closed_exact_recipes_forbidden':['NEXT4 A/C/E/G/H recipes','NEXT3 router and scale-repair exact variants','C205 lag subset variants']}
    atomic_json(out/'RESEARCH_POLICY_LOCK.json',policy,immutable=True)
    family={'status':'FROZEN_NEXT5_FAMILY_MENU','created_utc':utc_now(),'families':families,
        'maximum_families':7,'declared_slots':sum(len(f['slots']) for f in families.values()),
        'policy_sha256':digest(out/'RESEARCH_POLICY_LOCK.json'),
        'each_slot_requires_exact_feature_model_source_registration_before_first_score':True,
        'unused_or_failed_slots_remain_visible':True}
    atomic_json(ROOT/'HYPOTHESIS_FAMILY_LOCK_NEXT5.json',family,immutable=True)
    atomic_json(ROOT/'CANDIDATE_BUDGET_NEXT5.json',{'status':'FROZEN_NEXT5_BUDGET','created_utc':utc_now(),
        'families_max':7,'screens_max':35,'full_candidates_max':20,'shadow_finalists_max':3,'initial_slots':20,
        'family_slot_counts':{k:len(v['slots']) for k,v in families.items()},
        'new_adaptive_slots_require_prior_registered_distinct_hypothesis_and_unused_hard_budget':True,
        'reference_fit_accounting_separate_from_candidate_count':True,'all_actual_reference_fits_count_in_learner_total':True},immutable=True)
    reference={'status':'FROZEN_NEXT5_REFERENCE_TRAINING_PROTOCOL','created_utc':utc_now(),
        'purpose':'Fair comparison with no NEXT5 Shadow IDs in training or validation.',
        'legacy_deployment_bytes_unchanged':True,'legacy_full_DEV_OOF_used_for_selection':False,
        'primary_full_components':['J102','C205','B2_augmented_J','D2_expert'],'primary_folds':5,
        'screen_components':['J102','C205'],'screen_folds':[0,4],
        'reference_formulas':{'BL80':'.8 J102 + .2 C205','B2':'.8 B2_augmented_J + .2 C205','D2':'.9 BL80 + .1 D2_expert'},
        'learner_params':parameters,'weighting':'W3','constant_columns':'training rows only',
        'training_scope':'Only current ResearchPool outer training groups; no Shadow or old validation training rows',
        'ALT_and_H':'Fit J102/C205 once per needed locked split/fold; cache and reuse across nominated candidates. No candidate-specific split changes.',
        'full_ResearchPool_for_shadow':'Train the unchanged frozen reference recipe and finalists only after manifest freeze, before first Shadow access.',
        'old_fitted_models':'Historical scores and production/cold-runtime references only; do not give them extra-data training advantage in ResearchPool selection.',
        'exact_feature_and_execution_source_hashes_required_before_reference_fit':True}
    atomic_json(BASE/'references/POLICY.json',reference,immutable=True)
    progress('NEXT5 finite menus and decision gates frozen','Seven families, 20 initial slots, ResearchPool-only references and one-shot Shadow confirmation; no new scores yet')
    print('NEXT5 RESEARCH POLICY FROZEN: 7 families, 20 slots; no score-dependent gates',flush=True)


if __name__=='__main__':main()
