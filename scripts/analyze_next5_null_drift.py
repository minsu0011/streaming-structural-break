"""Post-hoc null-energy explanation on the unchanged four 16k cohorts."""
from pathlib import Path
import argparse
import csv
import hashlib
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.next5.artifacts import BASE, atomic_json, read_json, digest, utc_now, progress
from src.next5 import mixture, scale
from src.next5.null_drift import energy_boundary, evaluate, describe
from src.next5.synthetic import cohort_series, cohort_menu, cohort_key
from src.next5.synthetic_cache import update_input_hash
from src.next5.scheduler import register_job

SOURCES = ('scripts/analyze_next5_null_drift.py', 'src/next5/null_drift.py',
    'src/next5/mixture.py', 'src/next5/scale.py', 'src/next5/innovation.py',
    'src/next5/synthetic.py', 'src/next5/synthetic_cache.py', 'tests/test_next5_null_drift.py')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--register', action='store_true')
    args = parser.parse_args()
    parent = BASE / 'null_drift'
    if args.register:
        atomic_json(parent / 'POLICY_LOCK.json', {'created_utc': utc_now(),
            'sources_sha256': {name: digest(ROOT / name) for name in SOURCES},
            'menu_sha256': digest(BASE / 'synthetic/MENU_LOCK.json'),
            'execution_sha256': digest(BASE / 'synthetic/EXECUTION_LOCK.json'),
            'scenarios': ['N0', 'N1', 'N2', 'N3'], 'generating_seeds': 256,
            'horizons': [1024, 4096, 16384], 'variance_alternatives': scale.VARIANCES.tolist(),
            'channels': ['raw', 'conditional'], 'feature_banks': ['cumulative', 'sparse_starts'],
            'timing': 'Post-hoc explanation after first-fold E/P/R long-null observations; no prospective confirmation claim',
            'question': 'Does H-estimated innovation energy cause positive cumulative Gaussian working likelihood under unchanged synthetic nulls?',
            'identity': 'sum LR(v) = -n*log(v)/2 + sum(z^2)*(1-1/v)/2',
            'arithmetic_identity_rtol': 1e-10, 'arithmetic_identity_atol': 1e-7,
            'statistics': ['mean', 'median', 'q05', 'q95', 'fraction_strictly_positive'],
            'inputs': 'Replay only original long_null partition 2; each input hash must equal the original full-fold-0 reference cohort receipt',
            'new_fits': 0, 'new_predictions': 0, 'new_seed_draws': 0, 'new_generator_parameters': 0,
            'interpretation': 'Descriptive working-likelihood energy identity and fixed feature trajectories, not alarm probabilities, a new model, an ablation, or an anytime-valid guarantee',
            'promotion_authority': False}, immutable=True)
        register_job({'id': 'NULL_DRIFT_CONTRACTS', 'kind': 'validation', 'priority': 35,
            'argv': ['-m', 'pytest', 'tests/test_next5_null_drift.py', '-q',
                '--junitxml=artifacts/next5/null_drift/CONTRACTS.xml'],
            'description': 'Analytic LR sign boundary and exact unchanged feature snapshots through multiple ring wraps',
            'retry_safe': False})
        register_job({'id': 'NULL_DRIFT_DIAGNOSTIC', 'kind': 'validation', 'priority': 84,
            'depends_on_success': ['NULL_DRIFT_CONTRACTS', 'SYN_REF_full_0'],
            'argv': ['scripts/analyze_next5_null_drift.py'],
            'description': 'Explain frozen long-null working evidence with original generating seeds and no new predictions',
            'retry_safe': False})
        progress('Fixed null-energy explanation registered', 'Four original 256-seed cohorts, 1024/4096/16384 horizons, no candidate changes')
        return
    policy = read_json(parent / 'POLICY_LOCK.json')
    for name, sha in policy['sources_sha256'].items():
        assert digest(ROOT / name) == sha
    assert digest(BASE / 'synthetic/MENU_LOCK.json') == policy['menu_sha256']
    assert digest(BASE / 'synthetic/EXECUTION_LOCK.json') == policy['execution_sha256']
    menu = read_json(BASE / 'synthetic/MENU_LOCK.json')
    assert digest(ROOT / 'COUNTEREXAMPLE_SUITE_LOCK.json') == menu['original_generator_policy_sha256']
    lock = read_json(ROOT / 'COUNTEREXAMPLE_SUITE_LOCK.json')
    for name, sha in menu['generator_sources_sha256'].items():
        assert digest(ROOT / name) == sha
    specifications = {spec['scenario']: spec for spec in cohort_menu(menu, lock) if spec['kind'] == 'long_null'}
    assert set(specifications) == set(policy['scenarios'])
    out = parent / 'fixed_v1'
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / 'START.json', {'created_utc': utc_now(), 'policy_sha256': digest(parent / 'POLICY_LOCK.json')}, immutable=True)
    energy_rows, likelihood_rows, feature_rows, proofs = [], [], [], []
    maximum_identity_error = 0.
    boundaries = energy_boundary(scale.VARIANCES)
    for scenario in policy['scenarios']:
        spec = specifications[scenario]
        assert spec['seeds'] == 256 and spec['horizon'] == 16384 and not spec['paired'] and not spec['control']
        reference_path = BASE / 'synthetic/references/full/fold_0' / cohort_key(spec) / 'RESULTS.json'
        reference = read_json(reference_path)
        assert reference['status'] == 'COMPLETE' and reference['identity']['cohort'] == spec
        assert reference['identity']['execution_sha256'] == policy['execution_sha256']
        collected = [[] for _ in range(4)]
        hasher = hashlib.sha256()
        for seed in range(256):
            h, online, tau = cohort_series('long_null', scenario, seed, 16384, False, lock)
            assert tau == -1
            update_input_hash(hasher, seed, False, h, online, tau)
            values, error = evaluate(h, online, policy['horizons'])
            maximum_identity_error = max(maximum_identity_error, error)
            for target, value in zip(collected, values):
                target.append(value)
            if (seed + 1) % 32 == 0:
                progress('Frozen long-null energy replay', scenario + ' ' + str(seed + 1) + '/256 original seeds')
                print('NEXT5 NULL ENERGY', scenario, seed + 1, flush=True)
        assert hasher.hexdigest() == reference['input_sha256']
        energy, likelihood, cumulative, sparse = [np.asarray(values) for values in collected]
        for index, horizon in enumerate(policy['horizons']):
            common = {'scenario': scenario, 'cohort_kind': 'long_null', 'horizon': horizon,
                'source_input_sha256': hasher.hexdigest(), 'conditional_descriptive_only': True}
            for channel, name in enumerate(policy['channels']):
                energy_rows.append({**common, 'channel': name, 'metric': 'mean_clipped_standardized_energy',
                    **describe(energy[:, index, channel] / horizon)})
                for alternative, variance in enumerate(scale.VARIANCES):
                    likelihood_rows.append({**common, 'channel': name, 'variance': float(variance),
                        'zero_drift_energy_boundary': float(boundaries[alternative]),
                        'positive_drift_when': 'mean_energy_below_boundary' if variance < 1 else 'mean_energy_above_boundary',
                        'metric': 'cumulative_log_LR_per_observed_point',
                        **describe(likelihood[:, index, channel, alternative] / horizon)})
            for kind, values in [('cumulative', cumulative), ('sparse_starts', sparse)]:
                names = mixture.feature_names({'version': 'next5_fixed_mixture_v1', 'kind': kind})
                for column, name in enumerate(names):
                    feature_rows.append({**common, 'bank': kind, 'feature': name,
                        **describe(values[:, index, column])})
        proofs.append({'scenario': scenario, 'original_reference_receipt_sha256': digest(reference_path),
            'original_input_sha256': hasher.hexdigest(), 'all_original_inputs_matched': True})
    files = {}
    for name, rows in [('INNOVATION_ENERGY.csv', energy_rows), ('WORKING_LIKELIHOOD_DRIFT.csv', likelihood_rows),
                       ('FIXED_MIXTURE_FEATURES.csv', feature_rows)]:
        with (out / name).open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        files[name] = digest(out / name)
    atomic_json(out / 'RESULTS.json', {'created_utc': utc_now(), 'status': 'PASS',
        'policy_sha256': digest(parent / 'POLICY_LOCK.json'), 'files_sha256': files,
        'input_proofs': proofs, 'generating_seeds_per_scenario': 256, 'scenarios': len(proofs),
        'horizons': policy['horizons'], 'maximum_cumulative_LR_identity_absolute_error': maximum_identity_error,
        'feature_bank_innovation_channels_exactly_equal': True, 'energy_rows': len(energy_rows),
        'likelihood_rows': len(likelihood_rows), 'feature_rows': len(feature_rows),
        'new_fits': 0, 'new_predictions': 0, 'new_seed_draws': 0, 'real_data_read': False,
        'promotion_authority': False, 'limitation': policy['interpretation'],
        'source_sha256': digest(Path(__file__))}, immutable=True)
    progress('Fixed null-energy explanation complete', 'Exact original inputs, analytic cumulative LR identity, no new score or alpha guarantee')


if __name__ == '__main__':
    main()
