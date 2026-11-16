"""Refit an existing candidate on the same frozen folds without replacing evidence."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
import argparse, gc, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.features.config import CONFIG
from src.features.cache import read_fold
from src.features.streaming import feature_names
from src.features.historical_calibration import CalibrationPolicy
from src.models.calibrated import CalibratedBundle
from src.models.learners import ModelBundle
from src.validation.research import fit_fold, summarize
from src.validation.splits import FrozenSplit
from src.scoring.ts_auc import ts_auc
from src.utils.artifacts import write_json, sha256, utc_now
from scripts.run_real_research import admission


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--candidate', required=True); parser.add_argument('--tag', required=True)
    parser.add_argument('--rebuild-missing',action='store_true',help='Rebuild omitted caches and use packaged original fold models when OOF arrays are unavailable')
    args = parser.parse_args()
    if not args.tag.replace('_', '').isalnum(): raise ValueError('Simple tag required')
    admission(raw_directory=ROOT / 'data/raw')
    split = FrozenSplit(ROOT / 'artifacts/validation')
    state = json.loads((ROOT / 'artifacts/real/RESEARCH_STATE.json').read_text())
    result = next(r for r in state['results'] if r['experiment_id'] == args.candidate)
    spec = result['candidate']
    config = CONFIG.variant(**{k: tuple(v) if isinstance(v, list) else v for k, v in spec.get('feature_changes', {}).items()})
    representation = spec.get('representation', 'base')
    oof = ROOT / 'data/processed/oof' / args.candidate
    manifest_path=oof/'MANIFEST.json'
    if not manifest_path.exists() and args.rebuild_missing:manifest_path=ROOT/'artifacts/handoff/selected_oof_manifests'/args.candidate/'MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['split_sha256'] != split.digest: raise RuntimeError('Existing OOF split changed')
    cache = ROOT / manifest['cache_directory']
    if not (cache/'MANIFEST.json').exists() and args.rebuild_missing:
        if representation.startswith('ar8_arch1_'):
            from src.features.arch_cache import build_arch_cache
            from src.models.representations import calibration_policy
            rebuilt=build_arch_cache(ROOT/'data/raw',ROOT/'data/cache',split,config,calibration_policy(representation))
        elif representation=='cross_order_ar4_ar8_equal':
            from src.features.cross_order_cache import build_cross_order_cache
            rebuilt=build_cross_order_cache(ROOT/'data/raw',ROOT/'data/cache',split,config)
        elif representation=='base':
            from src.features.cache import build_cache
            rebuilt=build_cache(ROOT/'data/raw',ROOT/'data/cache',split=split,config=config)
        else:
            from src.models.representations import ar_prefix,ar_input_order,calibration_policy
            if ar_prefix(representation):
                from src.features.ar_cache import build_ar_cache
                rebuilt=build_ar_cache(ROOT/'data/raw',ROOT/'data/cache',split,config,calibration_policy(representation),input_order=ar_input_order(representation))
            else:
                from scripts.run_calibrated_research import calibrated_cache
                rebuilt=calibrated_cache(ROOT/'data/raw',split,calibration_policy(representation),config)
        if rebuilt.resolve()!=cache.resolve():raise RuntimeError('Rebuilt feature cache identity differs from archived provenance')
    cache_manifest = json.loads((cache / 'MANIFEST.json').read_text())
    if cache_manifest.get('status') != 'COMPLETE' or cache_manifest['split_sha256'] != split.digest:
        raise RuntimeError('Cached feature split is not complete and frozen')
    folds = []
    for k in range(5):
        if representation == 'base': frame = read_fold(cache, k, split=split, config=config)
        else:
            path = cache / f'fold_{k}.parquet'
            if sha256(path) != cache_manifest['fold_sha256'][str(k)]: raise RuntimeError('Calibrated feature cache changed')
            frame = pd.read_parquet(path)
        folds.append(frame)
    destination = ROOT / 'artifacts/real/cv_reproduction' / args.tag / args.candidate
    if (destination / 'RESULTS.json').exists(): raise RuntimeError('Existing reproduction result is immutable; use a new tag')
    destination.mkdir(parents=True, exist_ok=True)
    report = {'status': 'RUNNING', 'timestamp': utc_now(), 'candidate': args.candidate, 'specification': spec,
              'split_sha256': split.digest, 'cache_manifest_sha256': sha256(cache / 'MANIFEST.json'),
              'original_fold_scores': result['fold_scores'], 'folds': [], 'LOCAL_FINAL_SEAL_OPENED': False,
              'reduced_targets_used': False, 'purpose': 'Independent deterministic refit, same fixed folds; no new model selection'}
    started = time.perf_counter()
    from src.models.representations import representation_feature_names
    names=representation_feature_names(representation,config)
    for k, validation in enumerate(folds):
        start = time.perf_counter()
        model, prediction, score = fit_fold(folds, k, spec['model'], spec['families'], spec['sampling'],
                                            split=split, params=spec.get('params', {}), config=config)
        from src.models.representations import wrap_model
        model = wrap_model(model, representation, config)
        original_model_path=ROOT/'artifacts/models/real'/args.candidate/f'fold_{k}.joblib'
        if (oof/f'fold_{k}.npy').exists():
            original=np.load(oof/f'fold_{k}.npy',allow_pickle=False);reference_kind='original untouched OOF array'
        elif args.rebuild_missing:
            archived=type(model).load(original_model_path)
            original=archived.predict(validation.loc[:,names].to_numpy(dtype=np.float32)).astype(np.float32)
            reference_kind='packaged original fold model on reconstructed DEV features';del archived
        else:raise FileNotFoundError('Original OOF array unavailable; --rebuild-missing permits comparison with the archived fold model')
        if prediction.shape != original.shape: raise RuntimeError('Reproduction row count changed')
        path = destination / f'fold_{k}.joblib'; model.save(path); loaded = type(model).load(path)
        matrix = validation.loc[:, names].to_numpy(dtype=np.float32)
        reloaded_prediction = loaded.predict(matrix).astype(np.float32)
        row = {'fold': k, 'rows': len(prediction), 'ts_auc': score,
               'original_ts_auc': float(result['fold_scores'][k]), 'ts_auc_abs_diff': abs(score - result['fold_scores'][k]),
               'prediction_max_abs_diff': float(np.max(abs(prediction - original))),
               'different_float32_rows': int(np.count_nonzero(prediction != original)),
               'save_load_max_abs_diff': float(np.max(abs(reloaded_prediction - prediction))),
               'model_sha256': sha256(path), 'original_model_sha256': sha256(original_model_path),'reference_kind':reference_kind,
               'elapsed_seconds': time.perf_counter() - start}
        report['folds'].append(row); write_json(destination / 'CHECKPOINT.json', report)
        print('CV REPRODUCTION', args.candidate, 'fold', k, 'different_rows', row['different_float32_rows'], 'score', score, flush=True)
        del model, loaded, matrix, prediction, reloaded_prediction; gc.collect()
    report['elapsed_seconds'] = time.perf_counter() - started
    report['status'] = 'PASS' if all(r['different_float32_rows'] == 0 and r['save_load_max_abs_diff'] == 0 and r['ts_auc_abs_diff'] == 0 for r in report['folds']) else 'FAIL'
    report['total_rows'] = sum(r['rows'] for r in report['folds'])
    write_json(destination / 'RESULTS.json', report)
    print('CV REPRODUCTION', report['status'], args.candidate, report['total_rows'], flush=True)
    if report['status'] != 'PASS': raise SystemExit(1)


if __name__ == '__main__': main()
