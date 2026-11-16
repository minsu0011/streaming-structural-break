"""One frozen historical AR residual candidate with full development CV."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
import gc, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.features.config import CONFIG
from src.features.streaming import feature_names
from src.features.ar_residual_input import ARResidualCalibratedState, implementation_hash, AR_POLICY
from src.features.historical_calibration import CalibrationPolicy
from src.models.ar_calibrated import ARCalibratedBundle
from src.validation.adaptive import ResearchBudget, promotion_decision
from src.validation.splits import FrozenSplit
from src.validation.research import fit_fold, summarize
from src.validation.diagnostics import profile
from src.data.loader import load_training
from src.utils.artifacts import write_json, sha256, utc_now
from scripts.run_real_research import admission
from scripts.verify_engineering import benchmark

ANCHOR = 'M4_lightgbm_ABCD_S3_e3da86d90151'


def main():
    if not ResearchBudget(ROOT / 'artifacts/validation/REAL_SESSION.json').can_start(1200): raise RuntimeError('Final audit buffer protected')
    admission(raw_directory=ROOT / 'data/raw'); split = FrozenSplit(ROOT / 'artifacts/validation')
    state = json.loads((ROOT / 'artifacts/real/RESEARCH_STATE.json').read_text())
    anchor = next(r for r in state['results'] if r['experiment_id'] == ANCHOR); candidate = anchor['candidate']
    config = CONFIG.variant(**{k: tuple(v) if isinstance(v, list) else v for k, v in candidate['feature_changes'].items()})
    calibration = CalibrationPolicy(method='historical_median_mad'); version = implementation_hash(calibration, config)
    eid = f'M4_ar_residual_input_ABCD_S3_{version[:12]}'; result_path = ROOT / 'artifacts/real' / f'{eid}.json'
    if result_path.exists(): raise RuntimeError('Frozen AR residual candidate already evaluated')
    policy = {'anchor': ANCHOR, 'candidate': candidate, 'change': 'historical-only fixed AR(4) residual input mapping',
              'ar_filter': AR_POLICY, 'representation_hash': version, 'split_sha256': split.digest,
              'frozen_before_any_ar_residual_scores': True, 'maximum_initial_candidates': 1}
    policy_path = ROOT / 'artifacts/real/AR_RESIDUAL_INPUT_POLICY.json'
    if policy_path.exists() and json.loads(policy_path.read_text()) != policy: raise RuntimeError('AR residual policy changed')
    write_json(policy_path, policy)
    start = time.perf_counter(); names = feature_names(config)
    cache = ROOT / 'data/cache' / f'ar_residual_input_{version[:16]}'; cache.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads((ROOT / 'data/processed/oof' / ANCHOR / 'MANIFEST.json').read_text())
    source_cache = ROOT / source_manifest['cache_directory']
    manifest_path = cache / 'MANIFEST.json'; manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    series = {s.dataset_id: s for s in load_training(ROOT / 'data/raw', split=split)}
    if not (manifest.get('status') == 'COMPLETE' and manifest.get('representation_hash') == version and manifest.get('split_sha256') == split.digest):
        manifest = {'status': 'BUILDING', 'representation_hash': version, 'split_sha256': split.digest, 'ar_policy': AR_POLICY,
                    'source_cache_manifest_sha256': sha256(source_cache / 'MANIFEST.json'), 'seal_rows': 0, 'fold_sha256': {}}
        write_json(manifest_path, manifest)
        for fold in range(5):
            frame = pd.read_parquet(source_cache / f'fold_{fold}.parquet'); matrix = np.empty((len(frame), len(names)), dtype=np.float32)
            for sid, positions in frame.groupby('dataset_id', sort=False).indices.items():
                item = series[int(sid)]; stream = ARResidualCalibratedState(item.historical, config=config, policy=calibration)
                if not np.array_equal(frame.time_online.to_numpy()[positions], np.arange(len(item.online))): raise RuntimeError('AR residual cache online row alignment changed')
                for position, point in zip(positions, item.online): matrix[position] = stream.update_and_get(point)
            if not np.isfinite(matrix).all(): raise RuntimeError('Nonfinite AR residual feature')
            frame.loc[:, names] = matrix; path = cache / f'fold_{fold}.parquet'; frame.to_parquet(path, index=False)
            manifest['fold_sha256'][str(fold)] = sha256(path); write_json(manifest_path, manifest)
            print('AR RESIDUAL cache fold', fold, 'rows', len(frame), flush=True); del frame, matrix; gc.collect()
        manifest['status'] = 'COMPLETE'; write_json(manifest_path, manifest)
    for fold in range(5):
        if sha256(cache / f'fold_{fold}.parquet') != manifest['fold_sha256'][str(fold)]: raise RuntimeError('AR residual cache byte hash mismatch')
    folds = [pd.read_parquet(cache / f'fold_{fold}.parquet') for fold in range(5)]
    metadata = pd.read_parquet(ROOT / 'artifacts/audits/SERIES_METADATA.parquet', filters=[('dataset_id', 'in', sorted(series))])
    report = {'status': 'RUNNING', 'experiment_id': eid, 'timestamp': utc_now(), 'model': 'M4_ar_residual_input',
              'families': candidate['families'], 'sampling': candidate['sampling'], 'phase': 'ar_input_refinement',
              'representation': 'ar_residual_input_historical_median_mad', 'representation_hash': version,
              'anchor': ANCHOR, 'policy': policy, 'fold_scores': [], 'cv_rows': [], 'fractures': [], 'latencies': [],
              'save_load_max_abs_diff': 0.0, 'live_cached_prediction_max_abs_diff': 0.0,
              'LOCAL_FINAL_SEAL_OPENED': False, 'reduced_targets_used': False}
    for fold, frame in enumerate(folds):
        model, prediction, score = fit_fold(folds, fold, candidate['model'], candidate['families'], candidate['sampling'], split=split, params=candidate['params'], config=config)
        bundle = ARCalibratedBundle(model, calibration, config); path = ROOT / 'artifacts/models/real' / eid / f'fold_{fold}.joblib'
        bundle.save(path); loaded = ARCalibratedBundle.load(path)
        matrix = frame.loc[:, names].to_numpy(dtype=np.float32); np.testing.assert_array_equal(loaded.predict(matrix).astype(np.float32), prediction)
        first = series[int(frame.dataset_id.iloc[0])]; h = first.historical.astype(np.float32); online = first.online.astype(np.float32)
        stream = loaded.make_state(h); actual = np.array([loaded.predict_one(stream.update_and_get(point)) for point in online], dtype=np.float32)
        np.testing.assert_array_equal(actual, prediction[:len(online)])
        def factory():
            stream = loaded.make_state(h)
            return lambda point: loaded.predict_one(stream.update_and_get(point))
        latency = benchmark(factory, np.tile(online, max(1, 5000 // len(online) + 1))[:5000], repeats=3)
        latency['model_bytes'] = path.stat().st_size; report['latencies'].append(latency); report['fold_scores'].append(score)
        report['cv_rows'].append({'experiment_id': eid, 'model': report['model'], 'families': candidate['families'], 'sampling': candidate['sampling'], 'fold': fold, 'ts_auc': score})
        report['fractures'].extend([{'experiment_id': eid, 'fold': fold, **r} for r in profile(frame.target, prediction, frame.time_online, frame.dataset_id, metadata)])
        oof = ROOT / 'data/processed/oof' / eid; oof.mkdir(parents=True, exist_ok=True); np.save(oof / f'fold_{fold}.npy', prediction, allow_pickle=False)
        write_json(oof / 'MANIFEST.json', {'cache_directory': str(cache.relative_to(ROOT)), 'split_sha256': split.digest, 'feature_hash': version, 'wire_dtype': 'float32', 'ar_policy': AR_POLICY})
        write_json(ROOT / 'artifacts/real' / f'{eid}_CHECKPOINT.json', report)
        print('AR RESIDUAL', fold, score, latency['throughput_median_us_per_point'], 'us', flush=True)
        del model, bundle, loaded, matrix; gc.collect()
    report.update(summarize(report['fold_scores'])); report['status'] = 'COMPLETE'; report['elapsed_seconds'] = time.perf_counter() - start
    report['latency_us'] = float(np.median([r['throughput_median_us_per_point'] for r in report['latencies']]))
    report['versus_frozen_anchor'] = promotion_decision(report, anchor)
    write_json(result_path, report); print('AR RESIDUAL COMPLETE', eid, report['mean'], report['median'], report['worst_fold'], flush=True)


if __name__ == '__main__': main()
