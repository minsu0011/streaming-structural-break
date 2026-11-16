"""Freeze classifier-level stress comparisons after the learned families closed."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.next7.common import BASE, read_json, save, digest, utc_now, note, require_guard
from src.next6_extension.synthetic import CASES


def main():
    require_guard()
    usage = read_json(BASE / 'runtime/CANDIDATE_USAGE.json')
    save('CLASSIFIER_STRESS_PROTOCOL_LOCK.json', {
        'status': 'FROZEN', 'created_utc': utc_now(),
        'scope': 'Final robustness diagnostic of five already fitted NEXT7 definitions; no new recipe, fit, threshold search or promotion from synthetic data.',
        'NEXT7_definitions': ['M_EVENT', 'G2_EVENT', 'I4_EVENT_SCALE_TREE', 'G2_FULL', 'G3_PAIRWISE_FULL'],
        'references': ['M0', 'M3', 'AMPLITUDE_0_9_M0'], 'fitted_folds': list(range(5)),
        'selection': 'Two fixed memory mechanisms, best observed NEXT7 correction I4, qualified G2 and last ranking-objective G3. All chosen before classifier calibration or new stress outcomes.',
        'same_training_scope': 'Original RP outerTRAIN for each matched fold. NEXT7 heads use verified inner OOF. M3 remains the frozen NEXT6 outer model.',
        'calibration': {'partition': 30, 'H_length': 2048, 'cases': ['N0', 'N1', 'N2', 'N3'],
            'seeds_per_case': 256, 'horizon': 1024, 'pooled_series': 1024,
            'cut': '974th order statistic of pooled per-series maxima; strict score > cut.',
            'M0_M3': 'Reuse frozen NEXT6 model-specific cutoffs, with M0 maximum-by-maximum replay parity before evaluation.',
            'NEXT7': 'Compute each fixed fold/head cutoff using the exact same H and O calibration paths.',
            'amplitude_control_cut': 'Exactly .9 times that fold M0 cutoff. Alarm path must equal M0 when each uses its own cutoff.',
            'boundary': 'All five fold calibrations frozen before any new classifier stress evaluation.'},
        'evaluation': {'partition': 31, 'seeds_per_case': 512, 'cases': list(CASES), 'H_length': 2048,
            'null_horizons': [1024, 4096, 16384], 'other_horizons': [1024, 4096],
            'paired_controls': 'Same generator seed and historical values; interventions only after scheduled time.',
            'M3_reuse': 'Reuse already verified NEXT6 per-seed classifier metrics through4096. Replay M3 only for new16384 null horizon, asserting old prefix metric parity.',
            'existing_M0_parity': 'Every recalculated M0 metric through4096 matches frozen NEXT6 within1e-12.',
            'unavailable_M3_fields': 'Additional control-only metrics absent from old M3 receipts remain missing, never invented or imputed.'},
        'metrics': ['any_alarm', 'first_alarm', 'prebreak_alarm', 'postbreak_first_detection',
            'postbreak_any_detection', 'first_detection_delay', 'first_postbreak_alarm_delay',
            'recovery_delay', 'recovery_censored', 'paired_effect_after_128_max', 'paired_excess_alarm_after_128',
            'control_any_alarm_when_available', 'treatment_alarm_control_never_when_available'],
        'amplitude_control_secondary': 'Also report .9*M0 against the unscaled M0 cutoff, explicitly a shared-cut amplitude diagnostic; this cannot count as structural robustness.',
        'alarm_semantics': 'Threshold crossings of the final detector score. State-engine event memory tables are separate and cannot substitute for classifier alarms.',
        'target_semantics': 'Temporary structural changes remain target1 after recovery; isolated/burst/repeated shocks remain target0.',
        'probability_limit': 'Scores are uncalibrated ranking outputs; a95% pooled fixed-horizon maximum cutoff supplies no conditional-H or anytime probability guarantee.',
        'uncertainty': {'draws': 2000, 'counts': 'Reuse synthetic_stats/COUNTS_2000x512.npy',
            'unit': 'Whole generating seed, same counts across cases and all five frozen overlapping fitted folds.',
            'summary': 'Mean across fixed fitted folds inside each seed bootstrap draw; folds are not five independent samples.',
            'conditioning': 'Conditional on fitted models and fixed null cutoffs; no calibration refits, learner refits, extra draws or simultaneous coverage claim.',
            'eligible_support': 'Paired metric intervals use the common observed seed/fold support; undefined detections and recovery are reported, never set to zero.'},
        'budget_snapshot': usage, 'candidate_budget_increment': 0, 'new_fits': 0,
        'source_sha256': {n: digest(ROOT / n) for n in ('src/next2/counterexamples.py', 'src/next6_extension/synthetic.py')},
        'generator_lock_sha256': digest(ROOT / 'COUNTEREXAMPLE_SUITE_LOCK.json'),
        'old_calibration_receipts': {str(f): digest(ROOT / f'artifacts/next6_extension/calibration/fold_{f}/DONE.json') for f in range(5)},
        'old_evaluation_receipts': {f'{f}/{case}': digest(ROOT / f'artifacts/next6_extension/extended_synthetic/fold_{f}/{case}_SERIES.csv') for f in range(5) for case in CASES}})
    note('Classifier stress protocol frozen', 'Five existing NEXT7 definitions, matched frozen folds, own-calibrated and amplitude controls', 'F5_CAUSAL_NESTED_INTEGRATION')


if __name__ == '__main__':
    main()
