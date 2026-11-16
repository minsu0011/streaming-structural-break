"""Evaluate frozen late averages on the exact preexisting 20 stress fits."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.data import prepare_dev
from src.next.late_evaluation import assert_late_time, load_full_oof, combine_fold
from src.next.late_blend import LateBlendBundle
from src.next.paired_blend import SameBankBlendBundle
from src.next.references import verify_reference_sources
from src.next.progress import checkpoint
from src.scoring.ts_auc import ts_auc
from src.utils.artifacts import write_json, sha256, utc_now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', required=True)
    args = parser.parse_args()
    assert_late_time(ROOT)
    verify_reference_sources(ROOT)
    data = prepare_dev(ROOT)
    candidate, original, metadata, _, original_sources = load_full_oof(data, args.candidate)
    out = ROOT/'artifacts/next/robustness/candidates'/args.candidate
    out.mkdir(parents=True, exist_ok=True)
    identity = {'candidate': candidate, 'original_oof_sources': original_sources,
        'implementation_lock_sha256': sha256(ROOT/'configs/next_late_implementation_lock.json'),
        'policy_sha256': sha256(ROOT/'ROBUSTNESS_SPLIT_LOCK.json'),
        'training': 'Each of 20 paired component fits is reused exactly; no new fitting or weight selection.',
        'primary_gate_status_retained': original['status']}
    if (out/'POLICY.json').exists() and json.loads((out/'POLICY.json').read_text(encoding='utf-8')) != identity:
        raise RuntimeError('Late stress definition changed')
    if not (out/'POLICY.json').exists():
        write_json(out/'POLICY.json', identity)
    rows, summaries = [], []
    started = time.perf_counter()
    for split in ('ALT_1', 'ALT_2', 'ALT_3', 'LEAVE_ONE_H_REGIME_OUT'):
        folder = out/split
        folder.mkdir(exist_ok=True)
        for fold in range(5):
            path = folder/f'fold_{fold}.json'
            if path.exists():
                saved = json.loads(path.read_text(encoding='utf-8'))
                if saved['status'] != 'PASS' or any(sha256(folder/name) != digest for name, digest in saved['files_sha256'].items()):
                    raise RuntimeError('Completed late stress fold changed')
                if any(sha256(ROOT/name) != digest for name, digest in saved['component_sources'].items()):
                    raise RuntimeError('Frozen stress component changed')
                rows.append(saved)
                continue
            tick = time.perf_counter()
            model, prediction, indices, sources = combine_fold(data, candidate, 'stress', fold, split)
            reference_folder = ROOT/'artifacts/next/robustness/reference/PRIMARY'/split
            reference = json.loads((reference_folder/f'fold_{fold}.json').read_text(encoding='utf-8'))
            for name, digest in reference['files_sha256'].items():
                if sha256(reference_folder/name) != digest:
                    raise RuntimeError('Stress primary changed')
            coords = pd.read_parquet(reference_folder/f'fold_{fold}_coordinates.parquet')
            for key in ('dataset_id', 'time_online', 'target'):
                np.testing.assert_array_equal(coords[key].to_numpy(), metadata[key][indices])
            model.save(folder/f'fold_{fold}.joblib')
            cls = LateBlendBundle if 'late_blend' in candidate else SameBankBlendBundle
            loaded = cls.load(folder/f'fold_{fold}.joblib')
            sid = metadata['dataset_id'][indices[0]]
            historical, points, _ = data.get(sid)
            state = loaded.make_state(historical)
            live = np.asarray([loaded.predict_one(state.update(point)) for point in points], dtype=np.float32)
            np.testing.assert_array_equal(live, prediction[metadata['dataset_id'][indices] == sid])
            score = ts_auc(metadata['target'][indices], prediction, metadata['time_online'][indices])
            np.save(folder/f'fold_{fold}.npy', prediction, allow_pickle=False)
            np.save(folder/f'fold_{fold}_global_rows.npy', indices, allow_pickle=False)
            files = [f'fold_{fold}.joblib', f'fold_{fold}.npy', f'fold_{fold}_global_rows.npy']
            result = {'status': 'PASS', 'candidate': args.candidate, 'split_name': split, 'fold': fold,
                'ts_auc': score, 'reference_ts_auc': reference['ts_auc'], 'delta_vs_primary': score-reference['ts_auc'],
                'validation_series': len(np.unique(metadata['dataset_id'][indices])), 'validation_rows': len(indices),
                'component_sources': sources, 'files_sha256': {name: sha256(folder/name) for name in files},
                'live_prediction_exact': True, 'new_fits': 0, 'seal_rows': 0, 'reduced_new_uses': 0,
                'created_utc': utc_now(), 'elapsed_seconds': time.perf_counter()-tick}
            write_json(path, result)
            rows.append(result)
            print('LATE STRESS', args.candidate, split, fold, score, flush=True)
        selected = [row for row in rows if row['split_name'] == split]
        scores = [row['ts_auc'] for row in selected]
        summary = {'status': 'PASS', 'split_name': split, 'fold_scores': scores,
            'mean': float(np.mean(scores)), 'median': float(np.median(scores)), 'worst': min(scores),
            'mean_delta': float(np.mean([row['delta_vs_primary'] for row in selected]))}
        write_json(folder/'RESULTS.json', summary)
        summaries.append(summary)
    pd.DataFrame([{key: value for key, value in row.items() if key not in ('files_sha256', 'component_sources')} for row in rows]).to_csv(out/'FOLDS.csv', index=False)
    write_json(out/'RESULTS.json', {'status': 'PASS', 'candidate': args.candidate,
        'completed_component_fit_pairs': len(rows), 'new_refits': 0, 'promoted': False,
        'summaries': summaries, 'primary_gate_status_retained': original['status'],
        'created_utc': utc_now(), 'elapsed_seconds': time.perf_counter()-started,
        'seal_rows': 0, 'reduced_new_uses': 0})
    checkpoint(ROOT, args.candidate+'/stress', ['20 fixed stress averages'], ['late forensic and runtime checks'],
        artifacts=[str((out/'RESULTS.json').relative_to(ROOT))], best=original['full']['mean'])
    print('LATE STRESS COMPLETE', args.candidate, flush=True)


if __name__ == '__main__':
    main()
