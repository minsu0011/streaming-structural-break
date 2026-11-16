"""Two preregistered fixed averages of two strong, different small-tree learners."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import gc, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.models.calibrated import CalibratedBundle
from src.validation.splits import FrozenSplit
from src.validation.adaptive import ResearchBudget, promotion_decision
from src.validation.research import summarize
from src.validation.diagnostics import profile
from src.features.streaming import feature_names
from src.scoring.ts_auc import ts_auc
from src.data.loader import load_training
from src.utils.artifacts import write_json, sha256, utc_now
from scripts.verify_engineering import benchmark

ANCHOR = 'M4_lightgbm_ABCD_S3_df2601a8c737'
PARTNER = 'M3_histgb_ABCD_S3_fce4bafd492a'
WEIGHTS = (.25, .5)


def main():
    if not ResearchBudget(ROOT / 'artifacts/validation/REAL_SESSION.json').can_start(600):
        raise RuntimeError('New-experiment cutoff protected')
    split = FrozenSplit(ROOT / 'artifacts/validation')
    state = json.loads((ROOT / 'artifacts/real/RESEARCH_STATE.json').read_text())
    reference = next(r for r in state['results'] if r['experiment_id'] == ANCHOR)
    directory = ROOT / 'artifacts/real/fixed_model_blends'
    policy = {'anchor': ANCHOR, 'partner': PARTNER, 'partner_weights': list(WEIGHTS),
              'formula': 'float32((1-w)*float64(float32(anchor_probability)) + w*float64(float32(partner_probability)))',
              'why': 'LightGBM and HistGB have comparable DEV performance; fixed averaging tests model-error complementarity without fitted stacking or weight search.',
              'components_frozen': True, 'weights_frozen_before_blend_scores': True,
              'split_sha256': split.digest, 'source_sha256': sha256(__file__)}
    if (directory / 'RESULTS.json').exists(): raise RuntimeError('Fixed blend experiment already completed')
    if (directory / 'POLICY.json').exists() and json.loads((directory / 'POLICY.json').read_text()) != policy:
        raise RuntimeError('Fixed blend policy changed')
    write_json(directory / 'POLICY.json', policy)
    manifests = [json.loads((ROOT / 'data/processed/oof' / eid / 'MANIFEST.json').read_text()) for eid in (ANCHOR, PARTNER)]
    if manifests[0]['cache_directory'] != manifests[1]['cache_directory'] or any(m['split_sha256'] != split.digest for m in manifests):
        raise RuntimeError('Shared-feature blend requires identical cache coordinates and frozen split')
    cache = ROOT / manifests[0]['cache_directory']
    ids = split.frame.loc[split.frame.role == 'DEVELOPMENT_POOL', 'dataset_id'].tolist()
    metadata = pd.read_parquet(ROOT / 'artifacts/audits/SERIES_METADATA.parquet', filters=[('dataset_id', 'in', ids)])
    first_series = {}
    for series in load_training(ROOT / 'data/raw', split=split):
        fold = split.assignment(series.dataset_id)[1]
        if fold not in first_series: first_series[fold] = series
        if len(first_series) == 5: break
    results = {weight: {'experiment_id': f'M7_fixed_average_HistGB_weight_{str(weight).replace(".", "p")}',
                        'model': 'M7_fixed_average', 'families': 'ABCD', 'sampling': 'S3', 'phase': 'fixed_model_average',
                        'representation': reference['representation'], 'fold_scores': [], 'cv_rows': [], 'fractures': [],
                        'latencies': [], 'live_oof_max_abs_diff': 0.0, 'reloaded_component_oof_max_abs_diff': 0.0,
                        'components': [ANCHOR, PARTNER], 'partner_weight': weight, 'LOCAL_FINAL_SEAL_OPENED': False,
                        'reduced_targets_used': False} for weight in WEIGHTS}
    started = time.perf_counter()
    for fold in range(5):
        frame = pd.read_parquet(cache / f'fold_{fold}.parquet')
        models = [CalibratedBundle.load(ROOT / 'artifacts/models/real' / eid / f'fold_{fold}.joblib') for eid in (ANCHOR, PARTNER)]
        matrix = frame.loc[:, feature_names(models[0].feature_config)].to_numpy(dtype=np.float32)
        if models[0].feature_hash != models[1].feature_hash: raise RuntimeError('Blend representation mismatch')
        original = [np.load(ROOT / 'data/processed/oof' / eid / f'fold_{fold}.npy', allow_pickle=False) for eid in (ANCHOR, PARTNER)]
        for model, saved in zip(models, original): np.testing.assert_array_equal(model.predict(matrix).astype(np.float32), saved)
        series = first_series[fold]; h = series.historical.astype(np.float32); online = series.online.astype(np.float32)
        points = np.tile(online, max(1, 5000 // len(online) + 1))[:5000]
        for weight, result in results.items():
            prediction = ((1 - weight) * original[0].astype(np.float64) + weight * original[1].astype(np.float64)).astype(np.float32)
            def factory():
                stream = models[0].make_state(h)
                def predict_one(point):
                    features = stream.update_and_get(point)
                    probabilities = [float(np.float32(model.predict_one(features))) for model in models]
                    return float(np.float32((1 - weight) * probabilities[0] + weight * probabilities[1]))
                return predict_one
            detector = factory(); actual = np.array([detector(p) for p in online], dtype=np.float32)
            rows = frame.dataset_id.to_numpy() == series.dataset_id
            np.testing.assert_array_equal(actual, prediction[rows])
            latency = benchmark(factory, points, repeats=5)
            latency['model_bytes'] = sum((ROOT / 'artifacts/models/real' / eid / f'fold_{fold}.joblib').stat().st_size for eid in (ANCHOR, PARTNER))
            result['latencies'].append(latency)
            score = ts_auc(frame.target, prediction, frame.time_online); result['fold_scores'].append(score)
            eid = result['experiment_id']; result['cv_rows'].append({'experiment_id': eid, 'model': result['model'], 'families': 'ABCD', 'sampling': 'S3', 'fold': fold, 'ts_auc': score})
            result['fractures'].extend([{'experiment_id': eid, 'fold': fold, **row} for row in profile(frame.target, prediction, frame.time_online, frame.dataset_id, metadata)])
            oof = ROOT / 'data/processed/oof' / eid; oof.mkdir(parents=True, exist_ok=True); np.save(oof / f'fold_{fold}.npy', prediction, allow_pickle=False)
            write_json(oof / 'MANIFEST.json', {'cache_directory': str(cache.relative_to(ROOT)), 'split_sha256': split.digest, 'components': [ANCHOR, PARTNER], 'partner_weight': weight, 'wire_dtype': 'float32', 'formula': policy['formula']})
            print('FIXED AVERAGE', weight, 'fold', fold, score, 'latency_us', latency['throughput_median_us_per_point'], flush=True)
        del frame, models, matrix, original; gc.collect()
    for result in results.values():
        result.update(summarize(result['fold_scores'])); result['status'] = 'COMPLETE'
        result['latency_us'] = float(np.median([v['throughput_median_us_per_point'] for v in result['latencies']]))
        result['versus_frozen_component_anchor'] = promotion_decision(result, reference)
    write_json(directory / 'RESULTS.json', {'status': 'COMPLETE', 'timestamp': utc_now(), 'policy': policy,
        'elapsed_seconds': time.perf_counter() - started, 'results': list(results.values()),
        'deployment_status': 'Components and formula replay verified; only integrate a deployment artifact if final selection gates warrant it.'})
    print('FIXED AVERAGES COMPLETE', [(r['partner_weight'], r['mean'], r['median'], r['worst_fold']) for r in results.values()], flush=True)


if __name__ == '__main__': main()
