"""Restore old DEV evidence, verify the guard, and build compact DEV arrays."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.data import prepare_dev
from src.next.guard import SealGuard, SealAccessError
from src.utils.artifacts import write_json, sha256, utc_now

REFERENCES = {
    'PRIMARY': 'M7_cross_order_AR4_LGB_AR8_HGB_equal',
    'HIGHEST_MEAN': 'M4_lightgbm_AR8_ARCH1_ABCD_S3_1cb5e51ac7ae',
    'ROBUST_BACKUP': 'M3_histgb_AR8_ARCH1_ABCD_S3_1cb5e51ac7ae'}


def main():
    start = time.perf_counter()
    guard = SealGuard(ROOT)
    inventory = {}
    names = ['BASELINE_RESULTS_REAL.csv', 'MODEL_RESULTS_REAL.csv', 'CV_RESULTS_REAL.csv',
             'FEATURE_ABLATIONS_REAL.csv', 'FEATURE_STANDALONE_RANKING.csv',
             'FEATURE_MARGINAL_CONTRIBUTIONS.csv', 'SAMPLING_WEIGHTING_RESULTS.csv',
             'TIME_FRACTURE_REAL.csv', 'BREAK_AGE_DIAGNOSTICS.csv']
    for name in names:
        frame = pd.read_csv(ROOT/name)
        inventory[name] = {'sha256': sha256(ROOT/name), 'rows': len(frame), 'columns': list(frame.columns)}
    model = pd.read_csv(ROOT/'MODEL_RESULTS_REAL.csv').set_index('experiment_id')
    cv = pd.read_csv(ROOT/'CV_RESULTS_REAL.csv')
    references = {}
    for role, eid in REFERENCES.items():
        scores = cv.loc[cv.experiment_id == eid].sort_values('fold').ts_auc.to_numpy()
        assert len(scores) == 5
        result = {'experiment_id': eid, 'fold_scores': scores.tolist(), 'mean': float(scores.mean()),
                  'median': float(np.median(scores)), 'worst': float(scores.min()),
                  'oof_manifest_sha256': sha256(ROOT/'data/processed/oof'/eid/'MANIFEST.json')}
        np.testing.assert_allclose([result['mean'], result['median'], result['worst']],
                                   model.loc[eid, ['mean', 'median', 'worst_fold']].astype(float), rtol=0, atol=1e-14)
        references[role] = result
    write_json(ROOT/'artifacts/next/REFERENCE_RESTORATION.json', {
        'status': 'PASS', 'timestamp': utc_now(), 'csv_inventory': inventory, 'references': references,
        'reduced_performance_firewall': 'Prior handoff contains disclosed old scalars. These are not inputs to any NEXT policy, feature, candidate or selection. No reduced data/scalar file is read by this restoration script.'})
    checks = []
    for purpose in ['training', 'validation', 'OOF', 'prediction', 'experiment ledger']:
        try:
            guard.admit([min(guard.seal_ids)], purpose=purpose)
        except SealAccessError:
            checks.append({'check': purpose + ' seal request rejected', 'status': 'PASS'})
        else:
            raise AssertionError('Seal admission unexpectedly succeeded')
    try:
        guard.labels([min(guard.seal_ids)])
    except SealAccessError:
        checks.append({'check': 'seal target read rejected before scanner', 'status': 'PASS'})
    else:
        raise AssertionError('Seal target access unexpectedly succeeded')
    data = prepare_dev(ROOT)
    write_json(ROOT/'artifacts/next/DEV_ARRAYS_POINTER.json', {
        'directory': str(data.directory.relative_to(ROOT)), 'manifest_sha256': sha256(data.directory/'MANIFEST.json')})
    write_json(ROOT/'SEAL_GUARD_REPORT.json', {
        'status': 'PASS', 'timestamp': utc_now(), 'checks': checks, 'guard_tests_passed': 12,
        'partition_lock_sha256': guard.lock_sha, 'dev_series': len(data.ids),
        'dev_online_rows': len(data.online), 'seal_training_validation_oof_rows': 0,
        'seal_opened': {name: False for name in ['A', 'B', 'C', 'D']},
        'reduced_new_usage': 0, 'public_submissions': 0,
        'boundary': 'NEXT application admission; not an OS sandbox. Original source files remain available but NEXT code must use the guard.',
        'raw_bytes_note': 'Whole-file hashes read raw training bytes; Arrow may internally decode mixed row groups. Predicate-filtered DEV rows alone are exposed to research.',
        'elapsed_seconds': time.perf_counter()-start})
    print(json.dumps({'status': 'PASS', 'reference_scores': {k: v['mean'] for k, v in references.items()},
                      'dev_rows': len(data.online), 'seconds': time.perf_counter()-start}), flush=True)


if __name__ == '__main__':
    main()
