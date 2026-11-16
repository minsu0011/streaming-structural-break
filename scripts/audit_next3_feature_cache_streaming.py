"""Audit every cached new feature and existing held-out component prediction."""
import os
import sys
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import argparse
import hashlib
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, read_json, atomic_json, digest, utc_now, progress
from src.next3.frozen import candidate, J102
from src.next3.negative_bayes import NegativeBayesBundle, NAMES
from src.next2.io import design, model_class
from src.next.data import prepare_dev
from src.next.readonly_jit import disable_loaded_disk_caches

KINDS = ('S1_LOWER_SEMISCALE', 'S2_LOWER_QUANTILE', 'S3_NEGATIVE_BAYES', 'I2_HISTORICAL_RANK_PRODUCTS')


def protect_feature_cache():
    denied = []
    data_root = (ROOT/'data').resolve()
    def audit(event, args):
        if event == 'open':
            raw, mode, flags = args
            if not isinstance(raw, (str, bytes, os.PathLike)): return
            writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if writing and Path(os.fsdecode(raw)).resolve().is_relative_to(data_root):
                denied.append(str(raw)); raise RuntimeError('Feature audit may not build or mutate caches')
        elif event in ('os.remove', 'os.rmdir', 'os.mkdir', 'os.chmod', 'os.truncate', 'os.rename', 'os.link'):
            for raw in args[:2] if event in ('os.rename', 'os.link') else args[:1]:
                if isinstance(raw, (str, bytes, os.PathLike)) and Path(os.fsdecode(raw)).resolve().is_relative_to(data_root):
                    denied.append(str(raw)); raise RuntimeError('Feature audit may not mutate the data tree')
    sys.addaudithook(audit)
    return denied


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--candidate', choices=KINDS, required=True); args = parser.parse_args()
    hypothesis = BASE/'hypotheses/D003_FEATURE_CACHE_STREAMING.json'; hyp = read_json(hypothesis)
    assert digest(Path(__file__)) == hyp['script_sha256']
    out = BASE/'maintenance/feature_streaming'/args.candidate; out.mkdir(parents=True, exist_ok=False)
    report_path = BASE/'experiments'/args.candidate/'RESULTS.json'; report = read_json(report_path)
    assert report['status'] == 'COMPLETE'; scope = report['scope']; source = report_path.parent/scope
    identity = {'candidate': args.candidate, 'scope': scope, 'script_sha256': digest(Path(__file__)),
                'hypothesis_sha256': digest(hypothesis), 'candidate_result_sha256': digest(report_path)}
    atomic_json(out/'START.json', {'created_utc': utc_now(), 'identity': identity}, immutable=True)
    disabled_cache_writes = protect_feature_cache(); disable_loaded_disk_caches(); data = prepare_dev(ROOT)
    if args.candidate == 'S3_NEGATIVE_BAYES':
        spec, _ = candidate(J102); base, metadata, names, sources, global_rows = design(data, spec, scope)
        cache = ROOT/'data/next3/negative_bayes'/scope; manifest = read_json(cache/'MANIFEST.json')
        assert manifest['status'] == 'COMPLETE' and digest(cache/'features.npy') == manifest['sha256']
        assert digest(cache/'MANIFEST.json') == report['negative_feature_manifest_sha256']
        extra = np.load(cache/'features.npy', mmap_mode='r', allow_pickle=False)
        x = np.concatenate([base, extra], axis=1); del base
        names += list(NAMES); loader = NegativeBayesBundle; suffix = 'primary'
    else:
        spec = report['candidate']; x, metadata, names, sources, global_rows = design(data, spec, scope)
        loader = model_class(spec); suffix = 'complement' if args.candidate.startswith('I2_') else 'primary'
    assert names == report['feature_names'] and x.dtype == np.float32 and np.isfinite(x).all()
    models = {}; expected = np.full(len(global_rows), np.nan, dtype=np.float32); bindings = {}
    for record in report['fold_records']:
        fold = record['fold']; receipt_path = source/f'fold_{fold}.json'; receipt = read_json(receipt_path)
        assert receipt['status'] == 'PASS'
        for name, sha in receipt['files_sha256'].items(): assert digest(source/name) == sha
        rows = np.load(source/f'fold_{fold}_global_rows.npy', allow_pickle=False)
        local = np.flatnonzero(metadata['fold'] == fold)
        np.testing.assert_array_equal(global_rows[local], rows)
        expected[local] = np.load(source/f'fold_{fold}_{suffix}.npy', allow_pickle=False)
        models[fold] = loader.load(source/f'fold_{fold}_{suffix}.joblib')
        bindings[str(fold)] = digest(receipt_path)
    row_ids = metadata['dataset_id']; starts = np.r_[0, np.flatnonzero(row_ids[1:] != row_ids[:-1])+1]
    stops = np.r_[starts[1:], len(row_ids)]; ids = row_ids[starts]
    assert len(ids) == len(np.unique(ids)); data.guard.admit(ids, purpose='Audit previously fitted DEV feature cache')
    feature_hash = hashlib.sha256(); reference_hash = hashlib.sha256(); predicted_hash = hashlib.sha256(); expected_hash = hashlib.sha256()
    points = 0; prediction_points = 0; began = time.perf_counter(); counts = {}
    feature_model = next(iter(models.values()))
    for number, (sid, start, stop) in enumerate(zip(ids, starts, stops), start=1):
        h, o, _ = data.get(int(sid)); assert len(o) == stop-start
        fold = int(metadata['fold'][start]); assert np.all(metadata['fold'][start:stop] == fold)
        state = feature_model.make_state(h)
        actual = np.asarray([state.update(v).copy() for v in o], dtype=np.float32)
        if not np.array_equal(actual.view(np.uint32), x[start:stop].view(np.uint32)):
            atomic_json(out/'FAILURE.json', {'status': 'FAIL', 'identity': identity,
                'dataset_id': int(sid), 'reason': 'Cached and actual causal streaming feature bytes differ'}, immutable=True)
            raise AssertionError('Complete streaming feature cache parity failed')
        feature_hash.update(actual.tobytes()); reference_hash.update(x[start:stop].tobytes()); points += len(o)
        if fold in models:
            p = np.asarray(models[fold].predict(actual), dtype=np.float32)
            np.testing.assert_array_equal(p, expected[start:stop])
            predicted_hash.update(p.tobytes()); expected_hash.update(expected[start:stop].tobytes()); prediction_points += len(o)
            counts[fold] = counts.get(fold, 0)+1
        if number % 500 == 0:
            progress('Every-row scientific cache parity', f'{args.candidate}: {number} series, no new fit/score/cache', experiment='D003_FEATURE_CACHE_STREAMING')
            print('NEXT3 COMPLETE CACHE PARITY', args.candidate, number, round(time.perf_counter()-began, 1), flush=True)
    assert points == len(x) and feature_hash.hexdigest() == reference_hash.hexdigest()
    assert prediction_points == int(np.isfinite(expected).sum()) and predicted_hash.hexdigest() == expected_hash.hexdigest()
    assert not disabled_cache_writes; data.guard.check_state()
    atomic_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(), 'identity': identity,
        'series': len(ids), 'points': points, 'feature_columns': len(names), 'feature_entries': int(x.size),
        'feature_bytes_sha256': feature_hash.hexdigest(), 'heldout_component_prediction_points': prediction_points,
        'heldout_component_prediction_sha256': predicted_hash.hexdigest(), 'heldout_series_by_fold': counts,
        'fold_receipts_sha256': bindings, 'cached_design_sources': sources, 'new_fits': 0, 'new_quality_scores': 0,
        'seal_rows': 0, 'cache_writes_attempted': 0, 'elapsed_seconds': time.perf_counter()-began,
        'scope_note': 'Validate existing scientific evidence after finite-menu closure; no promotion or additional selection from these outputs.'}, immutable=True)


if __name__ == '__main__': main()
