"""Prospective finite NEXT9 protocol; no new prediction outcome is read."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.next9
from src.next9.common import BASE, atomic, read_json, digest, activity, utc_now


def main():
    assert read_json(BASE / 'PREFLIGHT_NEXT9.json')['status'] == 'PASS'
    assert read_json(BASE / 'INITIAL_CONTRACT_TEST_GATE_NEXT9.json')['status'] == 'PASS'
    with activity('Freeze finite movement, inference, comparison and evidence protocol before new outcomes', 'PROTOCOL', 'B1_COMMON_CURRENT_RANGE'):
        old = ROOT / read_json(BASE / 'SESSION_NEXT9.json')['previous_NEXT8_base']
        protocol = dict(status='FROZEN', timestamp=utc_now(), name='NEXT9_MOVEMENT_STATE_ORIENTATION',
          prior_base=old.relative_to(ROOT).as_posix(), prior_protocol_sha256=digest(old / 'NEXT8_PROTOCOL_LOCK.json'),
          seed=9101604, RP_series=6400, observed_row_stride=8, horizons=[1,4,16],
          normalization=read_json(BASE / 'PREFLIGHT_NEXT9.json')['feature_identity']['normalization'],
          separate_claims=['A_AUXILIARY_MOVEMENT', 'B_CURRENT_TARGET_ORIENTATION', 'C_MATCHED_TS_AUC_GAIN'],
          scope_count=20, selection_splits_per_scope=3, normal_and_early_separate=True,
          scope_definition='5 original outer procedures plus15 original inner procedures. Every candidate selection uses only3 inner/subinner heldout panels within that procedure TRAIN.',
          subinner_mapping='Exactly NEXT8_SUBINNER_V1|outer_fold|meta_inner|original_group hash ordering; not reseeded NEXT9 splits',
          early=dict(max_training_origin_and_auxiliary_target_time=199, minimum_other_series_validation_time=400,
                     H_and_unlabelled_online_warmup_allowed=True, late_label_updates=False),
          time_domains=[[0,9],[10,24],[25,49],[50,99],[100,199],[200,399],[400,None]],
          H_length_domains=[[0,1249],[1250,1749],[1750,2499],[2500,3999],[4000,None]],
          H_context_domains='TRAIN-only terciles of H lag1 and H tail3; frozen mapping, UNKNOWN for undefined support',
          states=dict(primary='S1_DIRECTION', families=['S1_DIRECTION','S2_DIRECTION_RUN','S3_DIRECTION_SLOPE','S4_REVERSAL'],
            pre_state_information='Exactly last16 H/Online F values ending at t-1. Current F[t] never accepted by peek API.',
            epsilon='Existing H-only h1 epsilon=max(.1*median absolute H40 lag1 difference,.02)',
            available='H length>=160 and finite16 seed/history coordinates. Do not require previous-level q20-q80 gap.',
            unknown='H<160 or unavailable history; abstain variable contribution',
            run_bins=['1','2-3','4+'], direction_run_capped_at=15, reversal_age_capped_at=15,
            run_and_age_limit='15 observed differences from16 values; last bin is censored, not exact episode age beyond history.',
            slope='Least-squares slope of last8 values; strong iff abs(slope)>epsilon_H',
            reversal='Opposite nonflat movement, bridging FLATs; recent iff observed age<=3; computed through t-1 only',
            sparse_S2_fallback='Within TRAIN, states with<64 original groups collapse to their S1 parent. Freeze mapping and apply unchanged to validation.'),
          movement_screen=dict(features=list(range(153)), break_labels_read=False, tau_read=False,
            primary_learner='Fixed ridge alpha10, equal original-series weights, standardized TRAIN-only columns; multioutput future feature and class indicators. Class scores clipped1e-6 and renormalized.',
            primary_design=['F[t-1]','F[t-2]','F[t-4]','F[t-8]','F[t-16]','slope8','deltaPrev','directionRunCapped15','reversalAgeCapped15'],
            auxiliary_targets='For h1/4/16: F[t+h], direction(F[t+h]-F[t]) with H-only horizon epsilon, and opposite nonflat terminal movement vs pre-current direction. Auxiliary targets never enter design.',
            primary_forecast_common_origins='All3 horizons must be available within same series and early cutoff; report lost endpoint support explicitly.',
            baselines=dict(D0A='F[t] persistence; stronger timing than strict-pre-current D1, explicitly reported',
              D0B='Normalized H40 median0; also retain original calibration center/spread provenance',
              D0C='Observed direction(F[t]-F[t-1]) persistence, TRAIN-only smoothing; not used as pre-current state',
              D1='Strict own-past design; current F excluded', D2='D1+six fixed H context quantities',
              D3='D2+log1p(observed online time)', D4='D3+certified prior stable context; NEXT8 has0, so initial aliasD3',
              D5='D3+unchanged NEXT8 mechanical_controls(feature) with current andlags1/4/16; explicit stronger current-input control'),
            first_screen_comparators=['D0A','D0B','D0C','D1'],
            detailed_comparators=['D0A','D0B','D0C','D1','D2','D3','D4','D5'],
            eligibility='All3 horizons evaluated. In>=2/3 selection panels, D1 has positive gain vs both D0A andD0B at>=2/3 horizons, no horizon has negative mean gain; normal andearly-late both required. At least64 heldout original groups per panel/horizon. Across panels, Holm-adjusted one-sided cluster tests vsbothbaselines pass at>=2 horizons in normal andearly. No forced candidates.',
            multiplicity='One procedure family of153features*3horizons*2baselines*2modes; Holm. Diagnostic class/reversal losses never used to pick a best horizon.',
            ranking='Among eligible, descending minimum across modes/horizons of mean group MSE gain vsHcenter divided by max(HcenterMSE,1e-8); featureID tie break',
            scope_candidate_freeze_before_labels=True, global_label_gate_requires_all20_scope_candidate_freezes=True,
            classification_limits='Beating persistence alone is MEAN_REVERSION_ONLY_OR_UNRESOLVED. No significance does not prove absent dynamics.'),
          orientation=dict(common_range_quantiles=[.1,.9], range_fit='TRAIN-only pairwise intersection, fixed on validation',
            minimum_groups_per_state=64, minimum_positive_and_negative_groups_per_state=32,
            minimum_common_range_width=1e-6, minimum_retained_fraction_per_state=.1,
            comparators=['O0','O1','O2_LEVEL','O3_MOVE','O4_MOVE_RUN','O5_MOVE_REV','O6_TABULAR','STATE_ONLY','F_ONLY'],
            fixed_nonlinearity=['F','abs(F)','positive(F)','negative(F)','F_squared'],
            linear_learner='LogisticRegression C=.1 max_iter250 tol1e-6, TRAIN standardization, group-weighted',
            tabular='HistGradientBoostingClassifier max_depth3 max_iter100 learning_rate.05 min_samples_leaf128 l2=10 noearlystopping seed9101604; receives same raw/state inputs as compared O5 orM6',
            slope_contrast='Opposite heldout signed current-F effects, positive matched effect-difference95%CI, TRAIN-frozen common range, jointly adjusted family. Significant vs nonsignificant alone never counts.',
            required_value='State/F interaction must improve heldout proper logloss and same-time ranking vsO1 andSTATE_ONLY; O6 same-input contrast retained separately.',
            rule_menu_selection='TRAIN inner/subinner only. M4 provisional rules require consistent pair andorientation in>=2/3 internal heldout panels; exact deployed rule fixed from TRAIN fit.',
            final_certificate='Same pair andoppositeorientation>=4/5 outer folds;>=2 supported time and2H domains; early-to-late retains pattern; A eligibility, commonrange,effectdifference,O1andSTATE_ONLY increment required. O6 equivalence/support distinguished from explicit-network superiority.',
            equivalence='Do not infer equivalence from nonsignificance; same-input noninferiority margin=.001 TS-AUC with paired lowerCI>=-.001; superiority requireslowerCI>0.',
            null_limit='DYNAMICS_ONLY orCONTEXT_ONLY require supported restricted interpretation; unsupported/uncertain B remains UNRESOLVED.'),
          transition=dict(variants=['T0_PRIOR','T1_SELF_MARKOV','T2_SELF_RUN_SLOPE_H','T3_SAME_INPUT_DEPTH3'],
            conditioning='Pre-current observed state, run, slope, H context only; no target update online',
            fit='TRAIN group-weighted transition law; Dirichlet.5 smoothing for table; fixed coefficients/classifier for richer law',
            summaries=['stateNow','stateDurationCapped15','transitionSurprise','reversalProbability','continuationProbability','stateEntropy','timeSinceReversalCapped15','signedMomentumStrength'],
            topology='SELF first, frozen small graph. Diagnostic fit on movement-eligible candidates; model branch only provisionally qualified orientation features.'),
          cross=dict(max_parents_per_feature=2,max_edges=8,
            destination='Provisionally movement-conditional candidate within this TRAIN scope',
            source='Frozen target-free candidates or certified stable context',
            control='Exact remove-parent against SELF_BASE and mechanical D5 with deduplicated columns',
            support='Same horizon positive whole-original-series95%CI,>=4/5 finalouter replication, early-late retained; inner-only edge selection before predictions',
            no_edges='Preserve zero, alias M6=M5; no relaxed thresholds'),
          models=dict(M0='Matched excluded-group BL80 .8J102+.2C205', M1='Movement-eligible currentF fixednonlinear',
            M2='M1+ownpast/slope/run/reversal withoutstateinteraction', M3='M2+movementstate main effects',
            M4='M3+TRAIN-qualified state-specific currentF interactions', M5='M4+qualified SELF transition probability/surprise',
            M6='M5+supportedcross oraliasM5',
            TABULAR_SAME_INPUT='Same raw/state/path inputs asM6, depth3 fixedtree',
            absent_branch='Explicit alias; eligible currentF controls stay active even if B-certification empty',
            M7='Fit fixed ridge-logistic correction to inner excluded-group predictions with BL80 fixed offset, C=.1; only when inner paired correction gain CI>0 and>=2/3 positive. Otherwise exact BL80 fallback. No blend grid.'),
          promotion=dict(mean_delta_minimum=.0012,paired95_lower_strictly_positive=True,positive_folds_minimum=4,
            worst_fold_delta_minimum=-.001, major_time_H_collapse_forbidden=True, runtime_causality_lineage_reproduction_required=True,
            runtime_median_ratio_max=3.,runtime_p99_ratio_max=5., candidate_weights_not_retuned_after_outer=True),
          uncertainty=dict(unit='Original dependency group, whole series; never individual points',bootstrap_draws=2000,
            outer_statistic='Equal mean oforiginal5 fold TS-AUC', confidence=.95,
            screening_CI='Group-mean paired loss with normalapprox and Holm; final important contrasts get fixed2000groupbootstrap.',
            role='Adaptive RP development, not independent hidden confirmation or posterior probability'),
          negative_controls=dict(backgrounds=['GAUSSIAN','AR_0.8','ARCH','STUDENT_T5'],
            interventions=['NONE','MEAN_SHIFT','SCALE_UP','SCALE_DOWN','DEPENDENCE_SHIFT','SINGLE_SHOCK','REPEATED_SHOCK'],
            seeds=[9101604,9101605,9101606], H_lengths=[512,2048], online_length=512,
            diagnostic_only='Known synthetic intervention metadata excluded from infer/pre-state;no realRP candidate selection from control outcomes'),
          implementation_freeze='Versioned source/parameter locks and data provenance before execution, mutable task queue only by explicit registration. Any software repair preserves failed evidence and uses fresh taskID/processidentity.',
          productive_research_minimum_seconds=36000,separate_final_audit_minimum_seconds=3600,
          final_delivery='All specified reports and10real-evidencefigures, fresh raw-freeZIP actual replay, Desktop exactly20importantfiles+matchingZIP+fullZIP',
          no_new_Shadow_Seal_reduced_public_cloud_values=True)
        atomic(BASE / 'NEXT9_PROTOCOL_LOCK.json', protocol)
        budget = dict(status='FROZEN', timestamp=utc_now(), target_free_features=153, detailed_per_scope_maximum=12,
          conditional_finalists_maximum=6, conflicting_8_vs6_resolved_to_stricter6=True,
          state_families_maximum=4, transition_variants_maximum=4, summaries_maximum=8,
          horizons=[1,4,16], cross_parents_maximum=2, cross_edges_maximum=8,
          full_scientific_models=['M0','M1','M2','M3','M4','M5','M6','M7','TABULAR_SAME_INPUT'],
          fixed_minor_coefficients_not_micro_grid=True,no_candidate_count_expansion_to_fill_time=True)
        atomic(BASE / 'NEXT9_CANDIDATE_BUDGET.json', budget)
        atomic(BASE / 'NEXT8_VISUAL_RESTORATION_NEXT9.json', dict(status='PASS', timestamp=utc_now(),
          figure_count=8, all_eight_prior_PNGs_actually_viewed=True,
          combined_prior_PDF_sha256=digest(old / 'FIGURES/NEXT8_EIGHT_FIGURES.pdf'),
          observations=['Zero selectedcrossedges shown without invented links','No certifiedconditionalcontrast, no selected variablebeta',
                        'Ownforecast improvement overpersistence does not byitself beatHcenter','Newmodels collapsed underemptyfeatureselection;BL80 retained'],
          no_new_RP_outcome_used=True))
        atomic(BASE / 'runtime/CURRENT_RESEARCH_STATUS.json', dict(phase='TARGET_FREE_SETUP',bottleneck='B9_NORMALIZATION_STATE_AVAILABILITY',
          target_free_candidates=None,certified_variable_count=None,cross_edge_count=None,inner_best=None,
          next_task='Qualify target-free loader and state cache; execute all153 movement screen in original nested TRAIN scopes'), immutable=False)
        print('NEXT9 protocol and finite candidate budget frozen before new outcomes')


if __name__ == '__main__':
    main()
