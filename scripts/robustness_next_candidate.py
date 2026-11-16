"""Fixed stress refits of a screened, fully evaluated NEXT research candidate.

Diagnostics cannot waive an already failed primary promotion gate.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
import argparse
import gc
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.data import prepare_dev
from src.next.candidate_io import candidate_design, fit_candidate, model_class, load_full_oof
from src.next.references import verify_reference_sources
from src.next.progress import checkpoint
from src.scoring.ts_auc import ts_auc
from src.utils.artifacts import sha256, write_json, utc_now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    data = prepare_dev(ROOT)
    verify_reference_sources(ROOT)
    candidate, prior, _, _, oof_sources = load_full_oof(data, args.candidate)
    outroot = ROOT/'artifacts/next/robustness/candidates'/args.candidate
    outroot.mkdir(parents=True, exist_ok=True)
    policy_path = ROOT/'ROBUSTNESS_SPLIT_LOCK.json'
    policy = json.loads(policy_path.read_text(encoding='utf-8'))
    regime_dir = ROOT/'artifacts/next/current_forensic'
    regime_lock = json.loads((regime_dir/'REGIME_ASSIGNMENT_LOCK.json').read_text(encoding='utf-8'))
    regime_file = regime_dir/'DEV_SIGNATURES.parquet'
    if sha256(regime_file) != regime_lock['files_sha256'][regime_file.name]:
        raise RuntimeError('Historical regime assignments changed')
    # Never supply signature/tau fields to feature builders, learner or split.
    regimes = pd.read_parquet(regime_file, columns=['dataset_id', 'historical_regime']).set_index('dataset_id').historical_regime.to_dict()
    data.guard.admit(regimes, purpose='candidate historical-regime stress partition')
    splits = [(f'ALT_{j+1}', {int(k): v for k, v in item['fold_by_dataset_id'].items()}) for j, item in enumerate(policy['alternates'])]
    splits.append(('LEAVE_ONE_H_REGIME_OUT', regimes))
    design, metadata, names, sources, global_rows = candidate_design(data, candidate)
    np.testing.assert_array_equal(global_rows, np.arange(len(global_rows)))
    original_order = np.concatenate([np.flatnonzero(metadata['fold'] == k) for k in range(5)])
    identity = {'candidate': candidate, 'feature_sources': sources, 'original_oof_sources': oof_sources,
        'robustness_policy_sha256': sha256(policy_path), 'regime_lock_sha256': sha256(regime_dir/'REGIME_ASSIGNMENT_LOCK.json'),
        'script_sha256': sha256(Path(__file__)), 'candidate_io_sha256': sha256(ROOT/'src/next/candidate_io.py'),
        'primary_gate_status_at_nomination': prior['status'],
        'nomination': 'Valid full-primary-CV candidate; diagnostics do not permit promotion after a failed primary gate.',
        'train_order': 'Stress fold, then original fold, then original dataset ID and online time; matches frozen reference stress fits.'}
    if (outroot/'POLICY.json').exists():
        if json.loads((outroot/'POLICY.json').read_text(encoding='utf-8')) != identity:
            raise RuntimeError('Stress nomination or code changed')
    else:
        write_json(outroot/'POLICY.json', identity)
    rows = []
    for split_name, assignment in splits:
        code = pd.Series(metadata['dataset_id']).map(assignment).to_numpy()
        if not np.isfinite(code).all() or not set(np.unique(code)).issubset(set(range(5))):
            raise RuntimeError('Invalid candidate stress partition')
        partitions = [original_order[code[original_order] == fold] for fold in range(5)]
        out = outroot/split_name
        out.mkdir(exist_ok=True)
        for fold in range(5):
            result_path = out/f'fold_{fold}.json'
            if result_path.exists():
                result = json.loads(result_path.read_text(encoding='utf-8'))
                if result['status'] != 'PASS' or any(sha256(out/name) != digest for name, digest in result['files_sha256'].items()):
                    raise RuntimeError('Completed candidate stress fold changed')
                rows.append(result)
                print('CANDIDATE STRESS RESUME', args.candidate, split_name, fold, flush=True)
                continue
            train = np.concatenate([partitions[k] for k in range(5) if k != fold])
            valid = partitions[fold]
            data.guard.partition(np.unique(metadata['dataset_id'][train]), np.unique(metadata['dataset_id'][valid]))
            refdir = ROOT/'artifacts/next/robustness/reference/PRIMARY'/split_name
            reference_result = json.loads((refdir/f'fold_{fold}.json').read_text(encoding='utf-8'))
            for name, digest in reference_result['files_sha256'].items():
                if sha256(refdir/name) != digest:
                    raise RuntimeError('Reference stress artifacts changed')
            coords = pd.read_parquet(refdir/f'fold_{fold}_coordinates.parquet')
            for name in ('dataset_id', 'time_online', 'target'):
                np.testing.assert_array_equal(coords[name].to_numpy(), metadata[name][valid])
            tick = time.perf_counter()
            model, prediction, detail = fit_candidate(data, candidate, design, metadata, names, train, valid)
            model.save(out/f'fold_{fold}.joblib')
            loaded = model_class(candidate).load(out/f'fold_{fold}.joblib')
            np.testing.assert_array_equal(loaded.predict(design[valid[:4096]]).astype(np.float32), prediction[:4096])
            sid = metadata['dataset_id'][valid[0]]
            h, points, _ = data.get(sid)
            state = loaded.make_state(h)
            streaming = np.asarray([loaded.predict_one(state.update(point)) for point in points], dtype=np.float32)
            np.testing.assert_array_equal(streaming, prediction[metadata['dataset_id'][valid] == sid])
            score = ts_auc(metadata['target'][valid], prediction, metadata['time_online'][valid])
            np.save(out/f'fold_{fold}.npy', prediction, allow_pickle=False)
            np.save(out/f'fold_{fold}_global_rows.npy', valid, allow_pickle=False)
            files = [f'fold_{fold}.joblib', f'fold_{fold}.npy', f'fold_{fold}_global_rows.npy']
            result = {'status': 'PASS', 'candidate': args.candidate, 'split_name': split_name, 'fold': fold,
                'ts_auc': score, 'reference_ts_auc': reference_result['ts_auc'], 'delta_vs_primary': score-reference_result['ts_auc'],
                'training_series': len(np.unique(metadata['dataset_id'][train])), 'validation_series': len(np.unique(metadata['dataset_id'][valid])),
                'validation_rows': len(valid), 'elapsed_seconds': time.perf_counter()-tick, 'timestamp': utc_now(),
                'seal_rows': 0, 'reduced_new_uses': 0, 'live_prediction_exact': True,
                'engineering': detail, 'files_sha256': {name: sha256(out/name) for name in files}}
            write_json(result_path, result)
            rows.append(result)
            pd.DataFrame([{k: v for k, v in r.items() if k not in ('files_sha256', 'engineering')} for r in rows]).to_csv(outroot/'FOLDS.csv', index=False)
            print('CANDIDATE STRESS', args.candidate, split_name, fold, score, round(result['elapsed_seconds'], 1), flush=True)
            del model, loaded, prediction
            gc.collect()
        current = [r for r in rows if r['split_name'] == split_name]
        summary = {'status': 'PASS', 'split_name': split_name, 'fold_scores': [r['ts_auc'] for r in current],
            'mean': float(np.mean([r['ts_auc'] for r in current])), 'median': float(np.median([r['ts_auc'] for r in current])),
            'worst': min(r['ts_auc'] for r in current), 'mean_delta': float(np.mean([r['delta_vs_primary'] for r in current]))}
        write_json(out/'RESULTS.json', summary)
        checkpoint(ROOT, args.candidate+'/'+split_name, [args.candidate+'/'+split_name], ['remaining_candidate_stress', 'new_sequential_hypotheses'],
            artifacts=[str((out/'RESULTS.json').relative_to(ROOT))], best=max(.611292686371357, prior['full']['mean']))
    summaries = [json.loads((outroot/name/'RESULTS.json').read_text(encoding='utf-8')) for name, _ in splits]
    write_json(outroot/'RESULTS.json', {'status': 'PASS', 'candidate': args.candidate, 'completed_refits': len(rows),
        'primary_gate_status_retained': prior['status'], 'promoted': False, 'summaries': summaries,
        'elapsed_seconds': time.perf_counter()-started, 'timestamp': utc_now(), 'seal_rows': 0, 'reduced_new_uses': 0})
    print('CANDIDATE STRESS COMPLETE', args.candidate, len(rows), flush=True)


if __name__ == '__main__':
    main()
