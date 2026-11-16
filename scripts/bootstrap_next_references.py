"""Paired 2,000-series-bootstrap forensic comparison of all frozen references."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from numba import set_num_threads
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.guard import SealGuard
from src.next.bootstrap import PreparedAUC, paired_summary
from src.next.references import REFERENCES, load_reference_fold, verify_reference_sources
from src.next.progress import checkpoint
from src.validation.group_bootstrap import draw_group_multiplicities, weighted_ts_auc_replicates
from src.utils.artifacts import sha256, write_json, utc_now


def main():
    set_num_threads(4)
    start = time.perf_counter()
    guard = SealGuard(ROOT)
    verify_reference_sources(ROOT)
    policy = json.loads((ROOT/'ROBUSTNESS_SPLIT_LOCK.json').read_text(encoding='utf-8'))['bootstrap']
    out = ROOT/'artifacts/next/reference_bootstrap'
    out.mkdir(exist_ok=True)
    report_path = out/'RESULTS.json'
    if report_path.exists():
        stored = json.loads(report_path.read_text(encoding='utf-8'))
        if stored['status'] == 'PASS' and stored['policy'] == policy:
            print('REFERENCE BOOTSTRAP already complete; no repeated computation', flush=True)
            return
        raise RuntimeError('Existing bootstrap run incompatible')
    scores = {role: [] for role in REFERENCES}
    correlations = []
    coordinate_sha = {}
    kernel_checks = []
    for fold in range(5):
        frame, primary, _ = load_reference_fold(ROOT, guard, 'PRIMARY', fold)
        code, counts = draw_group_multiplicities(frame.dataset_id.to_numpy(), policy['final_replicates'], policy['seed']+fold)
        coordinate_sha[str(fold)] = sha256(ROOT/'data/processed/oof'/REFERENCES['PRIMARY']/f'fold_{fold}.npy')
        for role in REFERENCES:
            if role == 'PRIMARY':
                prediction = primary
            else:
                other, prediction, _ = load_reference_fold(ROOT, guard, role, fold)
                pd.testing.assert_frame_equal(frame, other)
            prepared = PreparedAUC.create(frame.target, prediction, frame.time_online, code, counts.shape[1])
            tick = time.perf_counter()
            samples = prepared.evaluate(counts)
            elapsed = time.perf_counter()-tick
            reference = weighted_ts_auc_replicates(frame.target, prediction, frame.time_online, code, counts[:4])
            np.testing.assert_array_equal(samples[:4], reference)
            scores[role].append(samples)
            correlations.append({'fold': fold, 'candidate': role,
                'pearson_vs_primary': float(np.corrcoef(primary, prediction)[0, 1]),
                'mean_absolute_disagreement': float(np.mean(np.abs(primary.astype(float)-prediction))),
                'max_absolute_disagreement': float(np.max(np.abs(primary.astype(float)-prediction)))})
            kernel_checks.append({'fold': fold, 'role': role, 'exact_reference_max_abs_diff': 0.,
                                  'replicates': policy['final_replicates'], 'elapsed_seconds': elapsed})
            np.save(out/f'{role}_fold_{fold}.npy', samples, allow_pickle=False)
            print('PAIRED BOOTSTRAP', role, fold, round(elapsed, 2), 'seconds', flush=True)
        write_json(out/'PROGRESS.json', {'completed_folds': fold+1, 'timestamp': utc_now(), 'checks': kernel_checks})
    matrices = {k: np.asarray(v).T for k, v in scores.items()}
    rows = []
    for role in ['HIGHEST_MEAN', 'ROBUST_BACKUP']:
        rows.append({'reference': 'PRIMARY', 'candidate': role, 'scope': 'primary_5fold_all_time',
                     **paired_summary(matrices['PRIMARY'], matrices[role])})
    rows.append({'reference': 'ROBUST_BACKUP', 'candidate': 'HIGHEST_MEAN', 'scope': 'primary_5fold_all_time',
                 **paired_summary(matrices['ROBUST_BACKUP'], matrices['HIGHEST_MEAN'])})
    pd.DataFrame(rows).to_csv(ROOT/'PAIRED_BOOTSTRAP_RESULTS.csv', index=False)
    pd.DataFrame(correlations).to_csv(out/'OOF_CORRELATIONS.csv', index=False)
    write_json(report_path, {'status': 'PASS', 'timestamp': utc_now(), 'policy': policy,
        'paired_results': rows, 'kernel_checks': kernel_checks, 'reference_oof_sha256': coordinate_sha,
        'elapsed_seconds': time.perf_counter()-start, 'seal_rows': 0, 'reduced_new_uses': 0,
        'interpretation': 'Conditional on fixed adaptively selected DEV OOF predictions; neither refit uncertainty nor selection bias is removed.'})
    checkpoint(ROOT, 'current-reference-forensic', ['seal_partition', 'guard', 'DEV_arrays', 'reference_2000_bootstrap'],
               ['historical_regimes', 'mechanism_signatures', 'OOF_disagreement', 'whitening'],
               artifacts=['SEAL_GUARD_REPORT.json', 'PAIRED_BOOTSTRAP_RESULTS.csv'], best=.611292686371357)
    print(json.dumps(rows), flush=True)


if __name__ == '__main__':
    main()
