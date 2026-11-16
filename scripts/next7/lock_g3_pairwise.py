"""Register the sixth and final learned family before any G3 fit or score."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.next7.common import BASE, save, read_json, digest, utc_now, note, require_guard


def main():
    require_guard()
    usage = read_json(BASE / 'runtime/CANDIDATE_USAGE.json')
    assert usage['full_reserved'] == 14 and usage['learned_meta_reserved'] == 4 and usage['major_families_reserved'] == 5
    contract = {'status': 'FROZEN', 'created_utc': utc_now(), 'generation': 'G3',
        'candidate': 'G3_PAIRWISE_FULL', 'family': 'F6_TIME_MATCHED_RANK_CORRECTION',
        'parent': 'G2_FULL: same ten standardized qualified-memory/current/scale features and same BL80 offset',
        'one_structural_change': 'Replace pointwise W3 classification loss by pairwise logistic ranking loss within the exact online time strata used by TS-AUC.',
        'motivation': {'G2_FULL_RP_delta': -.000801305733398916,
            'G2_FULL_observed_scale_decrease_delta': .01879622857591523,
            'G2_FULL_weak_or_ambiguous_delta': -.0011839285344893203,
            'interpretation': 'Posthoc diagnostic strata suggest heterogeneous benefit; this motivates examining the loss, not a future/mechanism gate.'},
        'features': 'Exactly G2 cached ten features; no new interactions, features, tau, dataset ID, future length or mechanism labels.',
        'standardization': 'Exactly the G2 outerTRAIN row mean/std floor1e-4, standardized clip8; do not refit on sampled pairs or outer validation.',
        'score': 'sigmoid(logit(clip(BL80,1e-6,1-1e-6)) + standardized_features @ beta)',
        'base_offset_coefficient': 1, 'free_intercept': False,
        'pair_definition': 'One positive and one negative outerTRAIN observation at the same integer time_online; no cross-time pairs.',
        'pair_sampling': {'per_time_cap': 1024, 'seed': 20260908,
            'ordering': 'Ascending online times; each class retains the frozen inner OOF row order. One PCG64 default_rng stream reset to this same seed for each outer fold.',
            'small_strata': 'If n_positive*n_negative<=1024, enumerate all pairs exactly.',
            'large_strata': 'Draw1024 positive indices and1024 negative indices independently, uniformly with replacement; pair corresponding draws.',
            'weight_each_draw': '(n_positive*n_negative)/(number_of_pairs_sampled_at_time * total_eligible_training_pair_mass)',
            'no_pair_eligible_time': 'Omit from objective, exactly as zero-pair times do not contribute to TS-AUC.'},
        'objective': 'sum_pair(weight * log(1+exp(-(offset_positive-offset_negative + beta@(z_positive-z_negative))))) + .01/2 * ||beta||^2',
        'penalty': .01,
        'optimizer': {'name': 'L-BFGS-B', 'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'initial_beta': 'zeros'},
        'nested': 'Reuse exact original BL80 inner OOF, verified outerTRAIN-only membership. Both members of every pair exclude every outer validation group.',
        'state_engine': 'G2 admission and all G1 current-state sources stay frozen; no confirmation/scale/horizon changes.',
        'source_and_sampling_receipts': 'Freeze tested fitter implementation and pair-plan hashes before training; all five model/prediction artifacts frozen before G3 outer-score release.',
        'required_pre_fit_tests': ['finite-difference gradient', 'small-stratum exhaustive pair parity',
            'pair weights match TS-AUC pair mass', 'no cross-time or same-series opposite-label pair',
            'predict_one/batch parity and fixed source validation', 'zero within-time feature contrast has zero ranking gradient'],
        'diagnostics_only': {'exact_pair_gradient_times': [0, 16, 64, 128, 256, 512, 768],
            'purpose': 'Compare sampled and exact outerTRAIN objective/gradient at these prespecified times. No extra draws or sample-size changes based on the result.',
            'probability_calibration': 'Ranking correction may change calibration; no calibrated probability claim.'},
        'bootstrap': {'candidates': ['G3_PAIRWISE_FULL'], 'draws': '5000 if mean matched delta>=.0025, otherwise2000; fixed before drawing',
            'references': ['M0', 'G2_FULL'], 'whole_series': True, 'no_extension_after_interval': True},
        'promotion_gate': 'Unchanged NEXT7 initial promotion/generation/robustness gates, including latency and leakage.',
        'scope_limit': 'One full candidate, one new learned recipe, one sixth family. No penalty/cap/seed/blend search.',
        'stop_rules': ['Any scientific failure closes this objective candidate; do not tune it against outer results.',
            'After this family, learned-recipe budget is exhausted. Continue forensic, calibration, runtime and reproducibility work.'],
        'theory_source': 'https://www.ijcai.org/Proceedings/15/Papers/137.pdf',
        'theory_limit': 'Population surrogate consistency does not establish finite-sample gains for this dependent, restricted, regularized nested model.',
        'G2_result_sha256': digest(BASE / 'G2_OUTER_SCORES.json'),
        'fracture_sha256': digest(BASE / 'FRACTURE_RESULTS_NEXT7_G1_G2.csv'),
        'budget_after_reservation': {'full_reserved': 15, 'full_remaining': 1, 'learned_meta_reserved': 5,
            'learned_meta_remaining': 0, 'major_families_reserved': 6, 'major_families_remaining': 0}}
    save('G3_PROTOCOL_LOCK.json', contract)
    save('research_notes/PAIRWISE_OBJECTIVE_RATIONALE.md', '''# G3: align the correction loss with within-time ranking

The official TS-AUC counts positive-negative comparisons at the same online time.
G1/G2 correction heads instead minimize a weighted pointwise classification loss.
These empirical objectives differ even when both use logistic functions: the
ranking loss uses score differences and is invariant to a common score shift
within a time stratum. A finite, regularized feature model need not obtain the
same coefficients from the two objectives.

Gao and Zhou analyze sufficient conditions for AUC surrogate consistency and
discuss pairwise logistic loss in [their IJCAI2015 paper](https://www.ijcai.org/Proceedings/15/Papers/137.pdf).
This supports considering the loss family, not a guarantee for NEXT7. Our rows
are dependent within histories, the correction has ten restricted coordinates,
and the objective uses finite sampled pairs plus a fixed penalty.

For time t, let M_t=n_positive(t)*n_negative(t) and K_t be the fixed sampled pair
count. Each sampled pair receives weight M_t/(K_t*sum_s M_s). Uniform sampling
with replacement therefore gives an unbiased estimate of the empirical full
pairwise loss conditional on the frozen outerTRAIN data. This algebraic fact
does not make pairs independent histories, remove training selection, or turn
RP development results into untouched test evidence.

Only the objective changes. No observed mechanism label or future change age
enters training features. G2 current state, memory admission, ten features,
standardization, baseline offset, regularization coefficient and optimizer limits
remain fixed. Sample cap1024 and seed20260908 are set before any G3 outcome.
''')
    usage.update(full_reserved=15, learned_meta_reserved=5, major_families_reserved=6,
        G3_candidate='G3_PAIRWISE_FULL', updated_utc=utc_now())
    save('runtime/CANDIDATE_USAGE.json', usage, False)
    note('G3 time-matched ranking protocol frozen', 'Sixth family; last allowed learned recipe', 'F6_TIME_MATCHED_RANK_CORRECTION')


if __name__ == '__main__':
    main()
