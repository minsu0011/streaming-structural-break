"""Add paired controls and frozen classifier alarms to the same eight traces."""
from pathlib import Path
import os
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['NUMBA_CACHE_DIR'] = str(ROOT / 'artifacts/next7/numba_cache')
from src.next7.worker_environment import environment
os.environ.update(environment())
import numpy as np
from src.next7.common import BASE, read_json, save, digest, utc_now, activity, require_guard, note
from src.next7.benchmark_adapter import Factory, CLASSIFIERS
from src.next7.null_growth import QualifiedState
from src.next7.calibration import load_calibration, cutoff
from src.next6_extension.synthetic import generated
from scripts.next7.export_network_traces import COORDINATES
from scripts.next7.conditional_null_grid import atomic_npz


def main():
    require_guard()
    source_path = BASE / 'visuals/STATE_NETWORK_TRACES.json'
    data = read_json(source_path)
    selected = read_json(BASE / 'visuals/TRACE_SELECTION_LOCK.json')['selected']
    assert [r['id'] for r in selected] == [r['id'] for r in data['traces']]
    cal = load_calibration()
    classifier_cal = read_json(BASE / 'CLASSIFIER_CALIBRATION_FROZEN.json')
    assert classifier_cal['status'] == 'FROZEN'
    methods = (*CLASSIFIERS, 'AMPLITUDE_0_9_M0')
    cuts = {m: float(classifier_cal['thresholds'][0][classifier_cal['methods'].index(m)]) for m in methods}
    save('visuals/v3/EXTENSION_LOCK.json', {'status': 'FROZEN', 'created_utc': utc_now(),
        'old_trace_sha256': digest(source_path), 'unchanged_selection_sha256': digest(BASE / 'visuals/TRACE_SELECTION_LOCK.json'),
        'classifier_calibration_sha256': digest(BASE / 'CLASSIFIER_CALIBRATION_FROZEN.json'),
        'methods': methods, 'fitted_fold': 0, 'new_path_selection': False,
        'scope': 'Illustration and causal-prefix debugging only; same success/failure examples, no population estimates.',
        'classifier_alarms': 'Compute strict crossing before display rounding; use each frozen classifier own cutoff.',
        'source_sha256': digest(__file__)})
    suite = read_json(ROOT / 'COUNTEREXAMPLE_SUITE_LOCK.json')
    factories = {m: Factory(m) for m in CLASSIFIERS}
    checks = []
    numeric = {}
    with activity('Export paired controls and classifier alarms for dynamic network', 'VISUAL_FORENSIC', 'F5_CAUSAL_NESTED_INTEGRATION'):
        for tr in data['traces']:
            h, o, tau, info = generated(tr['case'], tr['seed'], partition=31, horizon=tr['horizon'], control=False, lock=suite)
            hc, oc, _, _ = generated(tr['case'], tr['seed'], partition=31, horizon=tr['horizon'], control=True, lock=suite)
            if 'H_suffix' in tr:
                h, hc = h[-tr['H_suffix']:], hc[-tr['H_suffix']:]
            np.testing.assert_array_equal(h, hc)
            np.testing.assert_array_equal(np.round(o.astype(float), 6), tr['raw'])
            state = QualifiedState(h, core_cut=cal['core_cut'], scale_cut=cutoff(cal, len(h)))
            actual = state.replay(o)
            control = QualifiedState(hc, core_cut=cal['core_cut'], scale_cut=cutoff(cal, len(hc))).replay(oc)
            for name, column in COORDINATES.items():
                np.testing.assert_array_equal(np.round(actual[:, column], 6), tr['values'][name])
            np.testing.assert_array_equal(actual[:info['scheduled']], control[:info['scheduled']])
            tr['control_raw'] = np.round(oc.astype(float), 6).tolist()
            tr['control_values'] = {name: np.round(control[:, column], 6).tolist() for name, column in COORDINATES.items()}
            tr['classifiers'] = {}
            cached = {}
            for method in methods:
                if method == 'AMPLITUDE_0_9_M0':
                    p, c = .9 * cached['M0'][0], .9 * cached['M0'][1]
                else:
                    state_p, state_c = factories[method].new_state(h), factories[method].new_state(hc)
                    p = np.array([state_p.predict_one(float(v)) for v in o])
                    c = p.copy() if np.array_equal(o, oc) else np.array([state_c.predict_one(float(v)) for v in oc])
                np.testing.assert_array_equal(p[:info['scheduled']], c[:info['scheduled']])
                assert np.isfinite(p).all() and np.isfinite(c).all()
                cached[method] = p, c
                alarms, control_alarms = p > cuts[method], c > cuts[method]
                if method == 'AMPLITUDE_0_9_M0':
                    np.testing.assert_array_equal(alarms, cached['M0'][0] > cuts['M0'])
                    np.testing.assert_array_equal(control_alarms, cached['M0'][1] > cuts['M0'])
                tr['classifiers'][method] = {'cut': cuts[method], 'score': np.round(p, 9).tolist(),
                    'control_score': np.round(c, 9).tolist(), 'alarm': alarms.astype(int).tolist(),
                    'control_alarm': control_alarms.astype(int).tolist()}
                numeric[f'{tr["id"]}_{method}_score'] = p
                numeric[f'{tr["id"]}_{method}_control'] = c
                checks.append({'trace': tr['id'], 'method': method, 'fitted_fold': 0, 'points': len(o),
                    'prefix_before_intervention_exact': True, 'own_cut_used': True})
            numeric[f'{tr["id"]}_qualified_state'] = actual
            numeric[f'{tr["id"]}_control_qualified_state'] = control
        data.update({'viewer_version': 3, 'classifier_methods': methods, 'classifier_fitted_fold': 0,
            'classifier_default': 'M0', 'paired_controls_available': True,
            'classifier_precision': 'Nine decimals displayed; exact boolean alarms computed before rounding.',
            'default_future_context': False, 'same_eight_examples_not_reselected': True,
            'classifier_calibration_sha256': digest(BASE / 'CLASSIFIER_CALIBRATION_FROZEN.json'),
            'raw_competition_data_included': False})
        save('visuals/v3/STATE_NETWORK_TRACES.json', data)
        atomic_npz(BASE / 'visuals/v3/EXACT_SYNTHETIC_CURVES.npz', **numeric)
        save('visuals/v3/DATA_EXPORT_DONE.json', {'status': 'PASS', 'created_utc': utc_now(),
            'checks': checks, 'same_state_coordinates_as_v2': True, 'eight_traces': len(data['traces']),
            'classifier_series_pairs': len(checks), 'new_models_or_thresholds': False,
            'trace_sha256': digest(BASE / 'visuals/v3/STATE_NETWORK_TRACES.json'),
            'exact_curves_sha256': digest(BASE / 'visuals/v3/EXACT_SYNTHETIC_CURVES.npz'),
            'extension_lock_sha256': digest(BASE / 'visuals/v3/EXTENSION_LOCK.json')})
    note('Paired classifier network traces exported', 'Same eight examples, exact prefix parity, original own-cut alarms, future display optional', 'F5_CAUSAL_NESTED_INTEGRATION')


if __name__ == '__main__':
    main()
