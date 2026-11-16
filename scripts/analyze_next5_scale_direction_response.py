"""Explain fixed synthetic scale responses without adding a scientific recipe."""
from pathlib import Path
import argparse
import csv
import hashlib
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.next5.artifacts import BASE, atomic_json, read_json, digest, utc_now, progress
from src.next5 import scale, robust_null
from src.next5.synthetic import cohort_series
from src.next5.synthetic_cache import update_input_hash
from src.next5.direction_diagnostic import project, summaries, COORDINATES
from src.next5.scheduler import register_job

SCENARIOS = ('V1', 'V2', 'V3', 'V4', 'CB_N3_070', 'CB_N3_050')
CANDIDATES = ('S1_DIRECTIONAL_SCALE', 'S2_STUDENT_SCALE', 'S4A_AUGMENTED_J_SCALE',
    'S4B_UNIFIED_SCALE_SPECIALIST', 'R2_CONDITIONAL_SCALE_PIT', 'X1_SCALE_PERSISTENCE', 'X2_SCALE_TAIL')
SOURCES = ('scripts/analyze_next5_scale_direction_response.py', 'src/next5/direction_diagnostic.py',
    'src/next5/scale.py', 'src/next5/robust_null.py', 'src/next5/innovation.py',
    'src/next5/synthetic.py', 'src/next5/synthetic_cache.py', 'tests/test_next5_direction_diagnostic.py')


def read_predictions(directory, inputs):
    receipt_path = directory / 'RESULTS.json'
    receipt = read_json(receipt_path)
    path = ROOT / receipt['prediction_file']
    assert receipt['status'] == 'COMPLETE' and path.resolve().is_relative_to(directory.resolve())
    assert digest(path) == receipt['prediction_file_sha256']
    inputs[receipt_path.relative_to(ROOT).as_posix()] = digest(receipt_path)
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    np.testing.assert_array_equal(arrays['synthetic_seed_index'], np.repeat(np.arange(256), 2))
    np.testing.assert_array_equal(arrays['control'], np.tile([False, True], 256))
    return arrays, receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--register', action='store_true')
    args = parser.parse_args()
    parent = BASE / 'scale_direction_response'
    if args.register:
        atomic_json(parent / 'POLICY_LOCK.json', {
            'created_utc': utc_now(), 'sources_sha256': {name: digest(ROOT / name) for name in SOURCES},
            'menu_sha256': digest(BASE / 'synthetic/MENU_LOCK.json'),
            'execution_sha256': digest(BASE / 'synthetic/EXECUTION_LOCK.json'),
            'scenarios': SCENARIOS, 'candidates': CANDIDATES, 'folds': list(range(5)),
            'generating_seeds': 256, 'horizon': 1024, 'coordinates': COORDINATES,
            'timing': 'Post-hoc diagnostic after S1/S2/S4B first-fold variance-decrease failures; not a prospective confirmation claim',
            'feature_banks': 'Complete frozen unified-scale 80 coordinates and H-energy-PIT 16 coordinates',
            'prediction_source': 'Already saved matched fitted-fold synthetic predictions, unchanged',
            'missing_lag_512': 'Not observed when onset=512 and horizon=1024; report 171 paired seeds rather than changing horizon or imputing',
            'new_fits': 0, 'new_prediction_runs': 0, 'new_seed_draws': 0, 'new_generator_parameters': 0,
            'feature_replays': 'Only unchanged synthetic inputs; hash must match original reference input receipt',
            'interpretation': 'Equal generating-seed descriptive responses; no confidence intervals or feature ablation, no promotion authority',
        }, immutable=True)
        register_job({'id': 'SCALE_DIRECTION_CONTRACTS', 'kind': 'validation', 'priority': 35,
            'argv': ['-m', 'pytest', 'tests/test_next5_direction_diagnostic.py', '-q',
                '--junitxml=artifacts/next5/scale_direction_response/CONTRACTS.xml'],
            'description': 'Explicit synthetic onset coordinates, paired support and right-censored lag handling', 'retry_safe': False})
        register_job({'id': 'SCALE_DIRECTION_RESPONSE', 'kind': 'validation', 'priority': 92,
            'depends_on_success': ['SCALE_DIRECTION_CONTRACTS', 'SYN_ALL_FITTED_SUMMARY'],
            'argv': ['scripts/analyze_next5_scale_direction_response.py'],
            'description': 'Post-hoc variance-response attribution with original fixed features and all matched fitted-fold cached predictions',
            'retry_safe': False})
        progress('Fixed scale-direction diagnostic frozen', 'Original six scenarios and 256 seeds; all five fitted folds, no new learner or gate')
        return
    policy = read_json(parent / 'POLICY_LOCK.json')
    for name, sha in policy['sources_sha256'].items():
        assert digest(ROOT / name) == sha
    assert digest(BASE / 'synthetic/MENU_LOCK.json') == policy['menu_sha256']
    assert digest(BASE / 'synthetic/EXECUTION_LOCK.json') == policy['execution_sha256']
    menu = read_json(BASE / 'synthetic/MENU_LOCK.json')
    lock = read_json(ROOT / 'COUNTEREXAMPLE_SUITE_LOCK.json')
    assert digest(ROOT / 'COUNTEREXAMPLE_SUITE_LOCK.json') == menu['original_generator_policy_sha256']
    for name, sha in menu['generator_sources_sha256'].items():
        assert digest(ROOT / name) == sha
    out = parent / 'complete_v1'
    out.mkdir(parents=True, exist_ok=False)
    inputs, feature_rows, prediction_rows, input_hashes = {}, [], [], {}
    banks = ((scale, {'version': 'next5_scale_unification_v1', 'kind': 'unified'}),
        (robust_null, {'version': 'next5_robust_null_v1', 'kind': 'energy_pit'}))
    names = tuple(name for module, settings in banks for name in module.feature_names(settings))
    assert len(names) == 96
    for scenario in SCENARIOS:
        paired = np.full((2, 256, len(COORDINATES), len(names)), np.nan)
        hasher = hashlib.sha256()
        for seed in range(256):
            onset = lock['tau_cycle'][seed % 3]
            prefixes = []
            for control in (False, True):
                h, o, tau = cohort_series('counterexample', scenario, seed, 1024, control, lock)
                assert tau == (-1 if control else onset)
                update_input_hash(hasher, seed, control, h, o, tau)
                values = np.column_stack([module.make_state(h, settings).replay(o) for module, settings in banks])
                prefixes.append(values[:onset].copy())
                paired[int(control), seed] = project(values, onset)
            np.testing.assert_array_equal(*prefixes)
            if (seed + 1) % 32 == 0:
                progress('Scale-direction feature response', scenario + ' ' + str(seed + 1) + '/256 original seeds')
                print('NEXT5 SCALE DIRECTION', scenario, seed + 1, flush=True)
        input_hashes[scenario] = hasher.hexdigest()
        feature_rows.extend({'scenario': scenario, 'scope': 'fixed_feature_bank_no_learner', **row}
            for row in summaries(paired[0], paired[1], names))
        for fold in range(5):
            key = 'counterexample__' + scenario
            reference, receipt = read_predictions(BASE / 'synthetic/references/full' / ('fold_' + str(fold)) / key, inputs)
            assert receipt['input_sha256'] == hasher.hexdigest()
            groups = [('REFERENCE', reference, ('BL80', 'B2', 'D2'))]
            for cid in CANDIDATES:
                values, candidate_receipt = read_predictions(BASE / 'synthetic/candidates' / cid / 'full' / ('fold_' + str(fold)) / key, inputs)
                assert candidate_receipt['input_sha256'] == hasher.hexdigest()
                groups.append((cid, values, ('prediction', 'expert')))
            for cid, values, roles in groups:
                projections = np.empty((2, 256, len(COORDINATES), len(roles)))
                for seed in range(256):
                    onset = lock['tau_cycle'][seed % 3]
                    for control in (False, True):
                        row = seed * 2 + int(control)
                        projections[int(control), seed] = project(np.column_stack([values[role][row] for role in roles]), onset)
                prediction_rows.extend({'scenario': scenario, 'candidate': cid, 'fold': fold,
                    'scope': 'full_RP_CV_artifact', **row} for row in summaries(projections[0], projections[1], roles))
            progress('Cached scale-direction probability response', scenario + ' all fixed candidates fold ' + str(fold))
    files = {}
    for name, rows in [('FEATURE_DIRECTION_RESPONSE.csv', feature_rows), ('PREDICTION_DIRECTION_RESPONSE.csv', prediction_rows)]:
        with (out / name).open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        files[name] = digest(out / name)
    atomic_json(out / 'RESULTS.json', {'created_utc': utc_now(), 'status': 'PASS',
        'policy_sha256': digest(parent / 'POLICY_LOCK.json'), 'input_receipts_sha256': inputs,
        'exact_original_synthetic_input_hashes': input_hashes, 'feature_rows': len(feature_rows),
        'prediction_rows': len(prediction_rows), 'files_sha256': files,
        'prebreak_feature_prefix_exact': True, 'new_fits': 0, 'new_prediction_runs': 0,
        'new_seed_draws': 0, 'real_data_read': False, 'promotion_authority': False,
        'limitation': policy['interpretation']}, immutable=True)
    progress('Scale-direction response diagnostic complete', 'Original paired inputs and all fixed fold artifacts, no candidate changes')


if __name__ == '__main__':
    main()
