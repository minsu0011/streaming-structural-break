"""Fixed alternate-split and leave-one-H-regime-out refits of old references."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
import gc
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.guard import SealGuard
from src.next.references import REFERENCES, load_reference_fold, verify_reference_sources
from src.next.progress import checkpoint
from src.validation.research import fit_fold, summarize
from src.features.config import CONFIG
from src.utils.artifacts import sha256, write_json, utc_now


def main():
    started = time.perf_counter()
    guard = SealGuard(ROOT)
    verify_reference_sources(ROOT)
    policy_path = ROOT/'ROBUSTNESS_SPLIT_LOCK.json'
    policy = json.loads(policy_path.read_text(encoding='utf-8'))
    regime_dir = ROOT/'artifacts/next/current_forensic'
    regime_lock = json.loads((regime_dir/'REGIME_ASSIGNMENT_LOCK.json').read_text(encoding='utf-8'))
    metadata_path = regime_dir/'DEV_SIGNATURES.parquet'
    if sha256(metadata_path) != regime_lock['files_sha256'][metadata_path.name]:
        raise RuntimeError('Regime assignment changed')
    # Read only identity and H-only grouping columns; mechanism/tau excluded.
    regimes = pd.read_parquet(metadata_path, columns=['dataset_id', 'historical_regime']).set_index('dataset_id').historical_regime.to_dict()
    guard.admit(regimes, purpose='historical-regime stress partition')
    partitions = [(f'ALT_{j+1}', {int(k): v for k, v in item['fold_by_dataset_id'].items()})
                  for j, item in enumerate(policy['alternates'])]
    partitions.append(('LEAVE_ONE_H_REGIME_OUT', regimes))
    config = CONFIG.variant(normalization='median_mad', scales=(5,20,160))
    rows = []
    for role in ['HIGHEST_MEAN', 'ROBUST_BACKUP', 'PRIMARY']:
        chunks = [load_reference_fold(ROOT, guard, role, k, features=True)[0] for k in range(5)]
        full = pd.concat(chunks, ignore_index=True)
        del chunks
        gc.collect()
        for split_name, assignment in partitions:
            code = full.dataset_id.map(assignment).to_numpy(dtype=np.int8)
            if np.any(code < 0) or np.any(code > 4):
                raise RuntimeError('Invalid stress assignment')
            frames = [full.loc[code == fold].reset_index(drop=True) for fold in range(5)]
            out = ROOT/'artifacts/next/robustness/reference'/role/split_name
            out.mkdir(parents=True, exist_ok=True)
            identity = {'reference': REFERENCES[role], 'split_name': split_name,
                'robustness_policy_sha256': sha256(policy_path), 'regime_lock_sha256': sha256(regime_dir/'REGIME_ASSIGNMENT_LOCK.json'),
                'old_split_sha256': guard.split.digest, 'seal_lock_sha256': guard.lock_sha,
                'reference_oof_manifest_sha256': sha256(ROOT/'data/processed/oof'/REFERENCES[role]/'MANIFEST.json'),
                'script_sha256': sha256(Path(__file__)),
                'fit_source_sha256': sha256(ROOT/'src/validation/research.py')}
            manifest_path = out/'MANIFEST.json'
            if manifest_path.exists():
                if json.loads(manifest_path.read_text(encoding='utf-8')) != identity:
                    raise RuntimeError('Refusing mismatched robustness resume')
            else:
                write_json(manifest_path, identity)
            for fold in range(5):
                result_path = out/f'fold_{fold}.json'
                if result_path.exists():
                    result = json.loads(result_path.read_text(encoding='utf-8'))
                    if result['status'] != 'PASS' or any(sha256(out/name) != digest for name, digest in result['files_sha256'].items()):
                        raise RuntimeError('Completed stress fold integrity failure')
                    rows.append(result)
                    print('STRESS RESUME', role, split_name, fold, flush=True)
                    continue
                train_ids = set().union(*(set(frame.dataset_id) for j, frame in enumerate(frames) if j != fold))
                valid_ids = set(frames[fold].dataset_id)
                guard.partition(train_ids, valid_ids)
                tick = time.perf_counter()
                name = {'PRIMARY': 'M7_cross_order_equal', 'HIGHEST_MEAN': 'M4_lightgbm', 'ROBUST_BACKUP': 'M3_histgb'}[role]
                model, prediction, score = fit_fold(frames, fold, name, 'ABCD', 'S3', split=guard.split, params={}, config=config)
                model.save(out/f'fold_{fold}.joblib')
                np.save(out/f'fold_{fold}.npy', prediction, allow_pickle=False)
                coords = frames[fold][['dataset_id', 'time_online', 'target']]
                coords.to_parquet(out/f'fold_{fold}_coordinates.parquet', index=False)
                files = [f'fold_{fold}.joblib', f'fold_{fold}.npy', f'fold_{fold}_coordinates.parquet']
                result = {'status': 'PASS', 'candidate': role, 'split_name': split_name, 'fold': fold,
                    'ts_auc': score, 'training_series': len(train_ids), 'validation_series': len(valid_ids),
                    'elapsed_seconds': time.perf_counter()-tick, 'timestamp': utc_now(),
                    'seal_rows': 0, 'reduced_new_uses': 0, 'files_sha256': {n: sha256(out/n) for n in files}}
                write_json(result_path, result)
                rows.append(result)
                pd.DataFrame([{k:v for k,v in r.items() if k != 'files_sha256'} for r in rows]).to_csv(ROOT/'artifacts/next/ROBUSTNESS_REFERENCE_RESULTS.csv', index=False)
                print('STRESS', role, split_name, fold, score, round(result['elapsed_seconds'], 1), 'seconds', flush=True)
                del model, prediction, coords
                gc.collect()
            current = [r['ts_auc'] for r in rows if r['candidate'] == role and r['split_name'] == split_name]
            write_json(out/'RESULTS.json', {'status': 'PASS', **summarize(current), 'identity': identity})
            checkpoint(ROOT, f'{role}/{split_name}', [f'{role}/{split_name}'],
                ['remaining_reference_stress', 'whitening_screen', 'sequential_evidence'],
                artifacts=[str((out/'RESULTS.json').relative_to(ROOT))], best=.611292686371357)
            del frames
            gc.collect()
        del full
        gc.collect()
    report = pd.DataFrame([{k:v for k,v in r.items() if k != 'files_sha256'} for r in rows])
    report.to_csv(ROOT/'artifacts/next/ROBUSTNESS_REFERENCE_RESULTS.csv', index=False)
    stress = report.loc[report.split_name == 'LEAVE_ONE_H_REGIME_OUT'].rename(columns={'fold': 'regime'})
    stress['evaluation'] = 'leave_one_historical_regime_out_refit'
    old = pd.read_csv(ROOT/'HISTORICAL_REGIME_RESULTS.csv')
    old = old.loc[old.evaluation != 'leave_one_historical_regime_out_refit']
    pd.concat([old, stress], ignore_index=True).to_csv(ROOT/'HISTORICAL_REGIME_RESULTS.csv', index=False)
    write_json(ROOT/'artifacts/next/robustness/REFERENCE_SUITE.json', {
        'status': 'PASS', 'completed_fold_fits': len(report), 'elapsed_seconds': time.perf_counter()-started,
        'summary': report.groupby(['candidate','split_name']).ts_auc.agg(['mean','median','min']).reset_index().to_dict('records'),
        'seal_rows': 0, 'reduced_new_usage': 0})
    print('REFERENCE STRESS COMPLETE', len(report), round(time.perf_counter()-started,1), 'seconds', flush=True)


if __name__ == '__main__':
    main()
