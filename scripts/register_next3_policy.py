"""Freeze thresholds, small hypothesis budgets and initial evidence priorities."""
from pathlib import Path
import sys
import json
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, atomic_json, read_json, utc_now, digest, progress
from src.next3.bottlenecks import record_board


def main():
    old = read_json(ROOT/'configs/next2/RESEARCH_POLICY_LOCK.json')
    selection = read_json(ROOT/'DEV_CANDIDATES_NEXT2.json')
    policy = {'version': 'NEXT3_V1', 'created_utc': utc_now(),
        'reference_sha256': digest(BASE/'preflight/FROZEN_REFERENCES.json'),
        'selection_sha256': digest(ROOT/'DEV_CANDIDATES_NEXT2.json'),
        'primary_reference': 'BL2_C205_DEPENDENCE_80',
        'reference_mean': selection['performance']['full']['mean'],
        'reference_folds': selection['performance']['full']['fold_scores'],
        'screen': {'ids': old['screen']['ids'], 'folds': [0, 4],
            'source_sha256': digest(ROOT/'configs/next2/RESEARCH_POLICY_LOCK.json'),
            'mean_delta_min': -0.001, 'each_fold_delta_min': -0.004,
            'reference': 'matched frozen J/C 80/20 models at identical screen train/validation coordinates',
            'gate_models': 'Any learned gate uses strictly grouped inner base OOF within outer screen training',
            'pruning': 'Compare fixed 80/20 blend using changed specialist; do not promote from standalone score alone',
            'rejected_screen': 'No full CV; bug correction preserves original failed receipt'},
        'performance': {'mean_delta_min': .0015, 'median_delta_min': 0., 'worst_fold_delta_min': -.001,
            'bootstrap_probability_positive_min': .90, 'ci_lower_positive_required': True,
            'alternate_positive_count_min': 2, 'each_H_matched_delta_min': -.01,
            'supported_mechanism_min_break_series': 100,
            'supported_mechanism_improvement_min': .002, 'supported_improvement_CI_lower_positive': True,
            'supported_mechanism_max_mean_loss': .005},
        'generation': {'requires_performance_and_engineering': True, 'mean_delta_min': .0025,
            'bootstrap_probability_positive_strict_min': .95, 'alternate_positive_count_min': 3,
            'worst_fold_delta_min': 0., 'new_independent_information_or_clear_robustness_repair': True,
            'especially_strong_mean_delta': .0035},
        'robustness': {'mean_delta_min': -.0008, 'worst_fold_delta_min': 0.,
            'SCALE_delta_vs_frozen_blend_min': .002, 'SCALE_CI_lower_positive': True,
            'ARCH_scale_down_delta_min_each': .01, 'ARCH_variance_factors': [.7, .5],
            'outlier_excess_alarm_delta_max': 0., 'no_supported_mechanism_mean_loss_below': -.005},
        'engineering': {'constant_state_required': True, 'streaming_parity_required': True,
            'direct_10M_seconds_ratio_max': 1.5, 'private_state_ratio_max': 2.,
            'reference_runtime': 'FROZEN_PERFORMANCE shared_blend_v1; repeated same-session benchmark',
            'optimization_claim': 'Same scientific predictions; actual measured cost decrease with uncertainty/measurement scope reported'},
        'bootstrap': {'replicates': 2000, 'final_generation_replicates': 5000,
            'seed_base': old['bootstrap']['seed_base'], 'unit': 'whole DEV series, paired conditional fixed OOF',
            'selection_adjusted': False},
        'nested_cross_fit': {'inner_groups': 'Existing original group fold partitions inside each outer training set; for ALT/H use the same original group labels within outer train',
            'minimum_inner_partitions': 3, 'outer_validation_used_for_training': False,
            'base_hyperparameters': 'Frozen J102 and C205 100-tree W3 recipes',
            'training_row_order': 'ascending frozen global row coordinates'},
        'finite_menu': {
            'router': ['R1_SCORE_GAP', 'R1_EVIDENCE_RATIO', 'R2_LOGISTIC_GATE', 'R3_DEPTH2_GATE'],
            'scale_repair': ['S1_LOWER_SEMISCALE', 'S2_LOWER_QUANTILE', 'S3_NEGATIVE_BAYES', 'S4_CONDITIONAL_RATIO_DOWN'],
            'dependence_pruning': ['P1_LAGS_1_2', 'P2_LAGS_4_8', 'P3_LOCAL_ONLY', 'P4_BAYES_ONLY'],
            'dependence_invariance': ['I1_SIGN_PRODUCTS', 'I2_HISTORICAL_RANK_PRODUCTS'],
            'orthogonal': ['O1_LINEAR_RESIDUAL'],
            'family_budget': 'At most three major hypothesis families per bottleneck; these small menus are ceilings, not mandatory fits'},
        'routing_weight_bounds': [0., .3], 'static_fine_weight_search': False,
        'forbidden_inputs': ['dataset_id', 'tau', 'future_length', 'future_suffix', 'fold', 'posthoc_mechanism'],
        'long_null_budget_fraction_max': .2,
        'saturation': {'minimum_duration_seconds': 7200, 'failed_independent_hypotheses_min': 10,
            'best_full_delta_max': .0007, 'requires_only_repeated_ideas_remain': True},
        'all_seals_unopened': True, 'reduced_new_usage': 0, 'public_submission_new': 0, 'cloud_submission_new': 0}
    atomic_json(ROOT/'configs/next3/RESEARCH_POLICY_LOCK.json', policy, immutable=True)
    table_names = ['MODEL_RESULTS_NEXT2.csv','CV_RESULTS_NEXT2.csv','PAIRED_BOOTSTRAP_NEXT2.csv',
        'ALT_SPLIT_RESULTS_NEXT2.csv','REGIME_RESULTS_NEXT2.csv','MECHANISM_RESULTS_NEXT2.csv',
        'DISTILLATION_RESULTS.csv','RUNTIME_PRUNING_RESULTS.csv','COMPLEMENT_STANDALONE_RESULTS.csv',
        'COMPLEMENT_CORRELATION.csv','COMPLEMENT_BLEND_RESULTS.csv','COUNTEREXAMPLE_RESULTS.csv',
        'LONG_NULL_RESULTS_NEXT2.csv','EXPERIMENT_LEDGER.csv']
    inventory = {}
    for name in table_names:
        frame = pd.read_csv(ROOT/name, low_memory=False)
        inventory[name] = {'sha256': digest(ROOT/name), 'rows': len(frame), 'columns': frame.columns.tolist()}
    atomic_json(BASE/'preflight/INHERITED_TABLES.json', inventory, immutable=True)
    specs = [
        ('B1', 'DEPENDENCE gain / SCALE loss', 'Fixed C205 weight gains dependence but reduces supported scale score.',
         'MECHANISM_RESULTS_NEXT2.csv', .003, .85, .9, .7, .7, .35, .25,
         'Pair-contribution attribution and two preregistered causal heuristic routes', 'Reject failed screens; no static weight microsearch', 'ACTIVE'),
        ('B2', 'Negative scale evidence downstream', 'Signed current/forecast channels retain decrease information; downstream use is weak in ARCH/tail.',
         'docs/NEXT2_SCALE_DOWN_MECHANISTIC_SYNTHESIS.md', .002, .8, .9, .5, .6, .7, .35,
         'Matched stage separation, sign/range/saturation, then distinct one-sided repairs', 'Stop candidate family when synthetic improvement lacks DEV support', 'ACTIVE'),
        ('B3', 'Specialist redundancy and scale sensitivity', 'C205 has 4 lags and 6 statistic families; minimal useful complement is unknown.',
         'COMPLEMENT_STANDALONE_RESULTS.csv', .002, .6, .8, .7, .5, .4, .15,
         'Four small lag/family ablations, then two invariant alternatives only if routing fails', 'Blend gain is authority; standalone improvement alone is insufficient', 'QUEUED'),
        ('B4', 'Adaptive DEV selection uncertainty', '8,000 DEV series have been reused across sessions; conditional bootstrap is not independent validation.',
         'LIMITATIONS_NEXT2.md', .004, .95, 1., .25, .15, .3, .1,
         'Freeze stronger gates and verify nested base OOF lineage plus limited negative controls', 'No seal access and no small-gain promotion', 'GUARDRAIL'),
        ('B5', 'Long-horizon accumulation', 'Eventual alarm FPR grows; decay harmed clean late detection despite rank parity.',
         'LONG_NULL_RESULTS_NEXT2.csv', .0005, .9, .8, .3, .4, .5, .6,
         'Only evaluate finalists/new memory changes; prioritize competition score', 'Do not spend over 20% of total run on alarm calibration', 'DEFERRED'),
        ('B6', 'Isolated outlier persistence', 'Large isolated outlier leaves long-lived evidence and worsens blend false alarms.',
         'COUNTEREXAMPLE_RESULTS.csv', .001, .8, .8, .5, .5, .4, .25,
         'Impulse response through 1,024 points with age eligibility separated from expiry', 'Minimal component change only after identified cause; DEV gates remain', 'QUEUED'),
        ('B7', 'Early evidence weakness', 'Early-age ranking is weak but overall contribution may be small.',
         'CV_RESULTS_NEXT2.csv', .0004, .5, .5, .3, .6, .4, .2,
         'Measure exact pair-mass contribution before any early-only hypothesis', 'No overall loss for early-only gain', 'DEFERRED'),
        ('B8', 'Shared runtime and deployment reliability', 'Dual ensemble still costs more than distilled fallback; NEXT3 adds lifecycle failure modes.',
         'RUNTIME_PRUNING_RESULTS.csv', .001, .95, .7, .8, .1, .4, .1,
         'Supervisor recovery/atomic writes tests and profiling of finalized inference', 'No scientific changes presented as engineering parity', 'MAINTENANCE')]
    items = []
    for sid,name,description,evidence,impact,confidence,severity,achievable,risk,cost,runtime,next_test,stop,status in specs:
        items.append({'id':sid, 'name':name, 'description':description,
            'evidence':[{'path':evidence,'sha256':digest(ROOT/evidence)}],
            'real_DEV_support':'Existing locked original/ALT/H evidence; adaptive selection limitation retained',
            'synthetic_support':'Fixed NEXT2 counterexample evidence, diagnostic only',
            'estimated_metric_impact':impact, 'confidence':confidence, 'unresolved_severity':severity,
            'achievable_improvement':achievable, 'overfit_risk':risk, 'engineering_cost':cost,
            'runtime_cost':runtime, 'next_test':next_test, 'stop_rule':stop, 'status':status})
    record_board(items, 'Initial priorities from directly restored NEXT2 artifacts; effect sizes are planning estimates', initial=True)
    progress('NEXT3 gates registered', 'Immutable screen/promotion thresholds and initial B1/B2 bottleneck board')
    print('NEXT3 POLICY AND INITIAL BOARD FROZEN', flush=True)


if __name__ == '__main__':
    main()
