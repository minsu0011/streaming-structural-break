"""Aggregate every controlled scenario without choosing a favorable setting."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.next8
from src.next8.common import BASE, read_json, digest, atomic, csv_write, utc_now, activity
from src.next8.gate_reachability import cases


def main():
    lock = read_json(BASE / 'GATE_REACHABILITY_IMPLEMENTATION_LOCK_NEXT8.json')
    for name, sha in lock['source_sha256'].items():
        assert digest(ROOT / name) == sha, name
    with activity('Interpret all sixteen controlled discovery gates', 'RESULT_INTERPRETATION', 'B1_CONDITIONAL_DIRECTION'):
        records, rows, sources = [], [], {}
        for index, (world, rho, seed) in enumerate(cases()):
            directory = BASE / 'gate_reachability' / ('case_' + format(index, '02d'))
            r = read_json(directory / 'DONE.json')
            assert r['status'] == 'COMPLETE' and (r['world'], r['noise_rho'], r['seed']) == (world, rho, seed)
            for k, sha in r['inner_DONE_sha256'].items():
                assert digest(directory / ('inner' + k) / 'DONE.json') == sha
            sources[str(index)] = digest(directory / 'DONE.json')
            records.append(r)
            row = r['selection']['classifications'][0]
            rows.append({'case_index': index, 'world': world, 'noise_rho': rho, 'seed': seed,
                         'core_gate_classification': row['classification'],
                         'invariant_inner_passes': sum(v['invariant_pass'] for v in r['inner_results']),
                         'nonlinear_inner_passes': sum(v['fixed_nonlinear_pass'] for v in r['inner_results']),
                         'conditional_repeated_inner_folds': row['conditional_supported_folds'],
                         'mean_fixed_nonlinear_AUC': row['mean_fixed_nonlinear_AUC'],
                         'mean_conditional_gain': row['mean_conditional_gain'],
                         'conditional_fitted_inner_scopes': sum(v['conditional']['status'] == 'COMPLETE' for v in r['inner_results']),
                         'statistical_fixture_fits': r['statistical_fixture_fits'],
                         'same_full153_multiple_testing_family': False,
                         'post_outer_diagnostic_not_confirmation': True})
        csv_write('GATE_REACHABILITY_CONTROLS_NEXT8.csv', rows)
        atomic(BASE / 'GATE_REACHABILITY_CONTROLS_NEXT8.json', {
            'status': 'COMPLETE', 'timestamp': utc_now(), 'case_count': len(records), 'cases': records,
            'case_DONE_sha256': sources, 'table_sha256': digest(BASE / 'GATE_REACHABILITY_CONTROLS_NEXT8.csv'),
            'statistical_fixture_fits': sum(r['statistical_fixture_fits'] for r in records),
            'competition_fits_predictions_scores': 0, 'models_or_feature_gates_changed': False,
            'interpretation': [
                'A passing artificial signal demonstrates reachability of the core screen under that specific law; it does not certify any actual RP feature.',
                'A failed positive control may reveal support, common-range, representation or repeated-inner limitations. It is not evidence that its generating signal is absent.',
                'Only two seeds per world/memory setting, one synthetic coordinate and population normal thresholds were used; do not estimate general power or error rates from these counts.',
                'This is a post-outer diagnostic and does not repair or retune the frozen NEXT8 predictive procedure.',
                'The real feature bank, H60/H40 engine, H-context second-stage selection and full153-feature multiplicity are outside this limited control.',
                'Validation fixture fits are separate from RP scientific-candidate fits and model-family counts.'
            ]})


if __name__ == '__main__':
    main()
