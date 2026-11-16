"""Reuse all fixed O1 trajectories; distinguish transient and persistent score effects."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next2.counterexamples import generate, policy
from src.next2.outlier_persistence import aligned_pair_predictions, paired_mean_interval
from src.validation.group_bootstrap import draw_group_multiplicities
from src.next2.session import checkpoint
from src.utils.artifacts import write_json, sha256, utc_now


def main():
    out = ROOT/'artifacts/next2/outlier_persistence'
    out.mkdir(parents=True, exist_ok=True)
    lock = policy()
    n = lock['evaluation_series_per_scenario']
    onsets = np.asarray([lock['tau_cycle'][i % len(lock['tau_cycle'])] for i in range(n)])
    offsets = [-1, 0, 1, 7, 31, 127, 255, 511]
    windows = [(0, 8), (8, 32), (32, 128), (128, 512)]
    identity = {'script_sha256': sha256(Path(__file__)),
        'source_sha256': sha256(ROOT/'src/next2/outlier_persistence.py'),
        'protocol_sha256': sha256(ROOT/'docs/NEXT2_OUTLIER_PERSISTENCE_PROTOCOL.md'),
        'original_suite_sha256': sha256(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json'),
        'seed': 2026090963, 'replicates': 2000, 'offsets': offsets, 'windows': windows,
        'alphas': [.05, .01], 'posthoc_descriptive': True, 'new_fits': 0}
    stored = out/'POLICY.json'
    # JSON converts tuples to lists; compare the serialized representation.
    canonical = json.loads(json.dumps(identity))
    if stored.exists():
        if json.loads(stored.read_text()) != canonical:
            raise RuntimeError('Outlier follow-up identity changed')
    else:
        write_json(stored, canonical)
    for i, onset in enumerate(onsets):
        hc, oc, tc = generate('O1', i, control=True, lock=lock)
        hi, oi, ti = generate('O1', i, control=False, lock=lock)
        assert tc == ti == -1
        np.testing.assert_array_equal(hc, hi)
        np.testing.assert_array_equal(np.flatnonzero(oc != oi), [onset])
    _, counts = draw_group_multiplicities(np.arange(n), 2000, identity['seed'])
    counts = counts[1:].astype(np.float64)
    sources = {}; models = []
    all_dev = ('J102_SERIAL_VARIANCE_512', 'Y001_LONG_VARIANCE_ARCH1',
               'B104_LONG_VARIANCE_NO_GLR_MEMORY', 'C205_DEPENDENCE', 'BL2_C205_DEPENDENCE_80')
    original_tables = [ROOT/'artifacts/next2/research_tables/ORIGINAL_COUNTEREXAMPLE_RESULTS.csv',
                       ROOT/'artifacts/next2/finalist_counterexamples/COUNTEREXAMPLES.csv']
    # Finalist summary location is fixed by its completed source receipt.
    for index, eid in enumerate(all_dev):
        base = ROOT/('artifacts/next2/counterexamples' if index < 3 else 'artifacts/next2/finalist_counterexamples')
        table = original_tables[0] if index < 3 else base/'COUNTEREXAMPLES.csv'
        models.append((eid, 'all_DEV', base/'counterexample/O1', eid+'.npy', table))
    for base in sorted((ROOT/'artifacts/next2/component_stress_cv0').iterdir()):
        if (base/'RESULTS.json').exists():
            models.append((base.name, 'original_CV_fold_0', base/'counterexample/O1', 'prediction.npy', base/'COUNTEREXAMPLES.csv'))
    assert len(models) == 30
    trajectories = []; occupancy = []
    for eid, fitted, folder, name, table_path in models:
        receipt = folder/'RESULTS.json'; result = json.loads(receipt.read_text())
        assert result['status'] == 'PASS'
        for filename in (name, 'synthetic_tau.npy'):
            assert sha256(folder/filename) == result['files_sha256'][filename]
        tau = np.load(folder/'synthetic_tau.npy', allow_pickle=False)
        assert tau.shape == (2*n,) and np.all(tau == -1)
        source = np.load(folder/name, allow_pickle=False)
        control, intervention = aligned_pair_predictions(source, onsets)
        sources[receipt.relative_to(ROOT).as_posix()] = sha256(receipt)
        sources[table_path.relative_to(ROOT).as_posix()] = sha256(table_path)
        common = {'candidate': eid, 'fitted_model': fitted, 'scenario': 'O1', 'seed_pairs': n}
        for offset in offsets:
            c, p = control[:, offset+1], intervention[:, offset+1]
            trajectories.append({**common, 'post_injection_offset': offset,
                'control_probability_mean': float(c.mean(dtype=np.float64)), 'outlier_probability_mean': float(p.mean(dtype=np.float64)),
                **{'paired_probability_delta_'+key: value for key, value in paired_mean_interval(p.astype(float)-c, counts).items()}})
        table = pd.read_csv(table_path, float_precision='round_trip')
        selected = table[(table.candidate == eid) & (table.scenario == 'O1')]
        for alpha in identity['alphas']:
            one = selected[selected.alpha == alpha]; assert len(one) == 1
            threshold = float(one.iloc[0].threshold)
            for start, end in windows:
                c = control[:, start+1:end+1] > threshold
                p = intervention[:, start+1:end+1] > threshold
                for metric, cv, pv in (('threshold_occupancy', c.mean(axis=1), p.mean(axis=1)),
                                       ('any_crossing_in_window', c.any(axis=1).astype(float), p.any(axis=1).astype(float))):
                    occupancy.append({**common, 'alpha': alpha, 'threshold': threshold,
                        'window_start_inclusive': start, 'window_end_exclusive': end, 'metric': metric,
                        'control': float(cv.mean()), 'outlier': float(pv.mean()),
                        **{'paired_delta_'+key: value for key, value in paired_mean_interval(pv-cv, counts).items()}})
        print('NEXT2 O1 PERSISTENCE', fitted, eid, flush=True)
    pd.DataFrame(trajectories).to_csv(out/'ALIGNED_PROBABILITIES.csv', index=False)
    pd.DataFrame(occupancy).to_csv(out/'WINDOW_ALARMS.csv', index=False)
    write_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(), 'identity': canonical,
        'models': len(models), 'pre_injection_prediction_exact': True,
        'generator_observations_differ_only_at_injection': True, 'input_receipts_sha256': sources,
        'files_sha256': {p: sha256(out/p) for p in ('ALIGNED_PROBABILITIES.csv', 'WINDOW_ALARMS.csv')},
        'limitations': 'Conditional fixed-model and fixed-calibration paired bootstrap. Descriptive follow-up after aggregate results; no selection adjustment. Continued score differences establish detector-state persistence, not causal attribution to a specific memory feature. O1 has no break; TS-AUC is undefined.',
        'new_fits': 0, 'seal_rows': 0, 'promotion': False})
    checkpoint('Single-observation score persistence and matched null contrasts', ['Scientific cost and mechanism synthesis', 'Final independent audit after six hours'])


if __name__ == '__main__':
    main()
