"""Check existing detector metrics under the queue's bounded BLAS environment."""
from pathlib import Path
import sys
import os
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.next7.worker_environment import environment, THREAD_KEYS
for key in THREAD_KEYS:
    os.environ[key] = environment()[key]
os.environ['NUMBA_CACHE_DIR'] = str(ROOT / 'artifacts/next7/numba_cache')
import numpy as np
from threadpoolctl import threadpool_info
from src.next7.common import BASE, read_json, save, digest, utc_now, activity, require_guard, note
from src.next7.classifier_stress import METHODS, predict_all, load_models, old_case_rows, curve_metrics, assert_old_parity
from src.next6_extension.synthetic import generated


def main():
    guard = require_guard()
    protocol = read_json(BASE / 'CLASSIFIER_STRESS_PROTOCOL_LOCK.json')
    models, _ = load_models(guard, include_m3=True)
    suite = read_json(ROOT / 'COUNTEREXAMPLE_SUITE_LOCK.json')
    checks = []
    with activity('Bounded math-thread classifier parity check', 'TEST_GATE', 'AUTONOMOUS_MAINTENANCE'):
        for case in ('N0', 'RETURN_NORMAL'):
            old = old_case_rows(case, protocol)
            h, o, tau, info = generated(case, 0, partition=31, horizon=4096, control=False, lock=suite)
            hc, oc, _, _ = generated(case, 0, partition=31, horizon=4096, control=True, lock=suite)
            p = predict_all(models, h, o, include_m3=True)
            control = p if np.array_equal(o, oc) else predict_all(models, hc, oc, include_m3=True)
            for fold in range(5):
                cal = read_json(ROOT / f'artifacts/next6_extension/calibration/fold_{fold}/DONE.json')['thresholds']
                for method in ('M0', 'M3'):
                    pos = METHODS.index(method)
                    for horizon in (1024, 4096):
                        actual = curve_metrics(p[fold, :, pos], control[fold, :, pos], cal[method], tau, info, horizon)
                        assert_old_parity(actual, old[fold, 0, method, horizon], horizon)
                        checks.append({'case': case, 'fold': fold, 'method': method, 'horizon': horizon, 'parity': 'PASS'})
        libraries = threadpool_info()
        assert all(r['num_threads'] == 1 for r in libraries if r['user_api'] == 'blas')
    save('CLASSIFIER_SINGLE_THREAD_PARITY_GATE.json', {'status': 'PASS', 'created_utc': utc_now(),
        'checks': checks, 'check_count': len(checks), 'threadpool_info': libraries,
        'math_environment_keys': {k: os.environ[k] for k in THREAD_KEYS},
        'source_sha256': digest(__file__), 'worker_environment_sha256': digest(ROOT / 'src/next7/worker_environment.py'),
        'old_metric_tolerance_unchanged': 1e-12, 'new_calibration_outcomes_released': False})
    note('Single-thread classifier parity gate passed', 'Original null and return-normal metrics match across five folds', 'F5_CAUSAL_NESTED_INTEGRATION')


if __name__ == '__main__':
    main()
