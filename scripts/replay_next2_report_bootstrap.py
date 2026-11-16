"""Recalculate reported best-model paired intervals from stored score draws, without raw rows."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.utils.artifacts import write_json, sha256, utc_now


def main():
    candidate = 'BL2_C205_DEPENDENCE_80'; reference = 'J102_SERIAL_VARIANCE_512'
    table = ROOT/'artifacts/next2/forensic'/candidate/'PAIRED_FRACTURES.csv'
    rows = pd.read_csv(table, float_precision='round_trip'); sources = {table.relative_to(ROOT).as_posix(): sha256(table)}
    checked = []
    for r in rows.itertuples():
        a, b = [], []
        for fold in range(5):
            for storage, eid, target in [('next2', candidate, a), ('next', reference, b)]:
                path = ROOT/'artifacts'/storage/'forensic'/eid/f'fold_{fold}_{r.scope}.npy'
                values = np.load(path, allow_pickle=False); assert values.shape == (2001,)
                sources[path.relative_to(ROOT).as_posix()] = sha256(path); target.append(values)
        a, b = np.asarray(a), np.asarray(b)
        means = a.mean(axis=0)-b.mean(axis=0)
        expected = {'candidate_mean': a[:, 0].mean(), 'reference_mean': b[:, 0].mean(),
            'estimate_mean_delta': means[0], 'ci_2_5': np.quantile(means[1:], .025),
            'ci_97_5': np.quantile(means[1:], .975), 'probability_delta_positive': np.mean(means[1:] > 0),
            'probability_delta_above_minus_0005': np.mean(means[1:] > -.0005)}
        for name, value in expected.items():
            assert np.isclose(value, getattr(r, name), rtol=0, atol=1e-14, equal_nan=True), (r.scope, name, value, getattr(r, name))
        checked.append({'scope': r.scope, 'fields_checked': list(expected)})
    assert len(checked) == 24
    write_json(ROOT/'artifacts/next2/report_bootstrap_replay/RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(),
        'script_sha256': sha256(Path(__file__)), 'source_sha256': sources, 'checks': checked,
        'bootstrap_replicates': 2000, 'scope_count': 24, 'new_fits': 0, 'raw_rows_read': 0,
        'method': 'Direct mean-of-fold resampling differences, percentile intervals and empirical strict-inequality fractions, rebuilt from stored per-fold score vectors. Fixed-fit conditional uncertainty only.'})
    print('NEXT2 BEST-MODEL BOOTSTRAP REPORT REPLAY PASS', len(checked), flush=True)


if __name__ == '__main__': main()
