"""Independent checkout/raw reconstruction, five-fold refit and all-DEV refit.

Only copied frozen specifications and checksums are used as references. The
original project caches, models and OOF arrays are inaccessible to this worker.
"""
import os
import sys
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
from pathlib import Path
import argparse
import gc
import hashlib
import importlib.metadata
import json
import time
from types import SimpleNamespace
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.guard import SealGuard
from src.next.data import iter_dev_series, RAW_SHA
from src.next.extensions import ExtendedFeatureState, extension_hash
from src.next.final_fitting import training_feature_names, fit_all_development
from src.next.candidate_io import fit_candidate, model_class
from src.data.labels import labels_from_tau
from src.scoring.ts_auc import ts_auc
from src.utils.artifacts import write_json, sha256, utc_now


def fitting_functions(candidate):
    if candidate.get('extension_registration'):
        from src.next.registered_final_fitting import training_feature_names as names, fit_all_development as fit
        return names, fit
    return training_feature_names, fit_all_development


def install_origin_access_guard(marker):
    origin = Path(marker['origin_project']).resolve()
    canonical_raw = Path(marker['canonical_raw']).resolve()
    environment = Path(sys.prefix).resolve()
    if ROOT == origin or ROOT != Path(marker['independent_workspace']).resolve():
        raise RuntimeError('This worker must execute in its marked independent checkout')
    counts = {'denied_origin_accesses': 0, 'canonical_raw_read_opens': 0}
    def under(path, parent):
        return path == parent or parent in path.parents
    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if not under(path, origin) or under(path, ROOT) or under(path, environment):
            return
        mode, flags = args[1:3]
        write = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (isinstance(flags, int) and bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
        if under(path, canonical_raw) and not write:
            counts['canonical_raw_read_opens'] += 1
            return
        counts['denied_origin_accesses'] += 1
        raise RuntimeError('Independent worker attempted to access the original project outside read-only canonical raw')
    sys.addaudithook(audit)
    return counts


def reconstruct_group(guard, candidate, names):
    identity = {'extension': candidate['extension'], 'extension_sha256': extension_hash(candidate['extension']),
        'names': list(names), 'split_sha256': guard.split.digest, 'seal_lock_sha256': guard.lock_sha,
        'raw_sha256': RAW_SHA, 'builder_sha256': sha256(Path(__file__)), 'dtype': 'float32',
        'construction': 'Fresh guarded raw iterator; ordinary scalar ExtendedFeatureState; no parent feature/OOF/model cache'}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    out = ROOT/'data/independent_next'/key[:20]
    out.mkdir(parents=True, exist_ok=True)
    path = out/'MANIFEST.json'
    if path.exists():
        record = json.loads(path.read_text(encoding='utf-8'))
        if record['identity'] != identity or record['status'] != 'PASS' or any(sha256(out/name) != digest for name, digest in record['files_sha256'].items()):
            raise RuntimeError('Independent reconstructed feature cache changed')
    else:
        started = time.perf_counter()
        matrices, targets = {}, {}
        for sid, historical, online, tau in iter_dev_series(guard):
            if sid in matrices:
                raise RuntimeError('Duplicate raw DEV series')
            state = ExtendedFeatureState(historical, names, candidate['extension'])
            matrix = np.empty((len(online), len(names)), dtype=np.float32)
            for t, point in enumerate(online):
                matrix[t] = state.update(point)
            matrices[int(sid)] = matrix
            targets[int(sid)] = labels_from_tau(len(online), tau)
            if len(matrices)%500 == 0:
                print('INDEPENDENT RAW FEATURES', key[:12], len(matrices), round(time.perf_counter()-started, 1), flush=True)
        if set(matrices) != set(guard.dev_ids):
            raise RuntimeError('Independent raw reconstruction lacks exact DEV coverage')
        ids = sorted(matrices)
        design = np.concatenate([matrices[sid] for sid in ids])
        metadata = {'dataset_id': np.concatenate([np.full(len(targets[sid]), sid, dtype=np.int32) for sid in ids]),
            'time_online': np.concatenate([np.arange(len(targets[sid]), dtype=np.int32) for sid in ids]),
            'target': np.concatenate([targets[sid] for sid in ids]),
            'fold': np.concatenate([np.full(len(targets[sid]), guard.split.assignment(sid)[1], dtype=np.int8) for sid in ids])}
        if len(design) != 4014405 or not np.isfinite(design).all():
            raise RuntimeError('Independent raw feature coverage or finiteness failure')
        del matrices, targets
        np.save(out/'features.npy', design, allow_pickle=False)
        files = ['features.npy']
        for name, values in metadata.items():
            np.save(out/(name+'.npy'), values, allow_pickle=False)
            files.append(name+'.npy')
        write_json(path, {'status': 'PASS', 'identity': identity, 'created_utc': utc_now(),
            'series': 8000, 'points': len(design), 'columns': len(names), 'elapsed_seconds': time.perf_counter()-started,
            'files_sha256': {name: sha256(out/name) for name in files}, 'seal_rows': 0})
        del design, metadata
        gc.collect()
    return np.load(out/'features.npy', mmap_mode='r', allow_pickle=False), {name: np.load(out/(name+'.npy'), mmap_mode='r', allow_pickle=False) for name in ('dataset_id', 'time_online', 'target', 'fold')}, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', default='configs/next_raw_reproduction_request.json')
    args = parser.parse_args()
    marker = json.loads((ROOT/'INDEPENDENT_NEXT_WORKSPACE.json').read_text(encoding='utf-8'))
    request_path = ROOT/args.request
    request = json.loads(request_path.read_text(encoding='utf-8'))
    if sha256(request_path) != marker['request_sha256']:
        raise RuntimeError('Independent reproduction request changed')
    access = install_origin_access_guard(marker)
    for relative, digest in marker['copied_input_sha256'].items():
        if sha256(ROOT/relative) != digest:
            raise RuntimeError('Copied frozen reproduction input changed')
    for name, digest in RAW_SHA.items():
        if sha256(ROOT/'data/raw'/name) != digest:
            raise RuntimeError('Canonical raw bytes changed')
    for module in list(sys.modules.values()):
        if module is not None and getattr(module, '__name__', '').startswith('src.'):
            path = Path(module.__file__).resolve()
            if ROOT not in path.parents:
                raise RuntimeError('Project code was imported outside the independent checkout')
    versions = {name: importlib.metadata.version(name) for name in request['numerical_versions']}
    if versions != request['numerical_versions']:
        raise RuntimeError('Independent numerical environment differs from the frozen environment')
    guard = SealGuard(ROOT)
    data = SimpleNamespace(guard=guard)
    groups = {}
    for item in request['candidates']:
        candidate = item['candidate']
        names = fitting_functions(candidate)[0](candidate)
        key = json.dumps(candidate['extension'], sort_keys=True)
        groups.setdefault(key, []).append((item, names))
    outroot = ROOT/'artifacts/next/raw_reproduction'
    outroot.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    candidates_done = []
    for items in groups.values():
        anchor, union_names = max(items, key=lambda pair: len(pair[1]))
        if any(not set(names).issubset(union_names) for item, names in items):
            raise RuntimeError('A shared raw reconstruction anchor does not cover all requested names')
        full_design, metadata, manifest = reconstruct_group(guard, anchor['candidate'], union_names)
        for item, names in items:
            candidate = item['candidate']
            eid = candidate['experiment_id']
            out = outroot/eid
            out.mkdir(exist_ok=True)
            if (out/'RESULTS.json').exists():
                record = json.loads((out/'RESULTS.json').read_text(encoding='utf-8'))
                if record['status'] != 'PASS' or record['candidate'] != candidate:
                    raise RuntimeError('Completed raw candidate reproduction changed')
                candidates_done.append(record)
                continue
            columns = [union_names.index(name) for name in names]
            design = np.asarray(full_design[:, columns], dtype=np.float32)
            fold_records = []
            for fold in range(5):
                path = out/f'fold_{fold}.json'
                if path.exists():
                    record = json.loads(path.read_text(encoding='utf-8'))
                    if record['status'] != 'PASS' or any(sha256(out/name) != digest for name, digest in record['files_sha256'].items()):
                        raise RuntimeError('Completed independent CV fold changed')
                    fold_records.append(record)
                    continue
                tick = time.perf_counter()
                train = np.concatenate([np.flatnonzero(metadata['fold'] == k) for k in range(5) if k != fold])
                valid = np.flatnonzero(metadata['fold'] == fold)
                model, prediction, details = fit_candidate(data, candidate, design, metadata, names, train, valid)
                model.save(out/f'fold_{fold}.joblib')
                loaded = model_class(candidate).load(out/f'fold_{fold}.joblib')
                np.testing.assert_array_equal(loaded.predict(design[valid[:4096]]).astype(np.float32), prediction[:4096])
                np.save(out/f'fold_{fold}.npy', prediction, allow_pickle=False)
                expected = item['original_folds'][str(fold)]
                actual_prediction_hash = sha256(out/f'fold_{fold}.npy')
                score = ts_auc(metadata['target'][valid], prediction, metadata['time_online'][valid])
                equal = actual_prediction_hash == expected['prediction_sha256']
                score_equal = abs(score-expected['ts_auc']) <= 1e-14
                files = [f'fold_{fold}.joblib', f'fold_{fold}.npy']
                record = {'status': 'PASS' if equal and score_equal else 'FAIL', 'candidate': eid, 'fold': fold,
                    'created_utc': utc_now(), 'points': len(valid), 'ts_auc': score,
                    'all_float32_predictions_bitwise_equal_by_sha256': equal, 'score_equal': score_equal,
                    'model_bytes_equal': sha256(out/f'fold_{fold}.joblib') == expected['model_sha256'],
                    'original_reference': expected, 'elapsed_seconds': time.perf_counter()-tick,
                    'files_sha256': {name: sha256(out/name) for name in files}, 'seal_rows': 0}
                write_json(path, record)
                if record['status'] != 'PASS':
                    raise RuntimeError('Independent raw CV predictions differ; retained failed artifacts require investigation')
                fold_records.append(record)
                print('INDEPENDENT RAW CV PASS', eid, fold, score, record['model_bytes_equal'], flush=True)
                del model, loaded, prediction
                gc.collect()
            final_path = out/'ALL_DEV.json'
            if final_path.exists():
                final = json.loads(final_path.read_text(encoding='utf-8'))
                if final['status'] != 'PASS' or sha256(out/'all_dev.joblib') != final['model_sha256']:
                    raise RuntimeError('Completed independent all-DEV model changed')
            else:
                model, details = fitting_functions(candidate)[1](design, metadata, candidate, guard, names)
                model.save(out/'all_dev.joblib')
                prediction = np.asarray(model.predict(design), dtype=np.float32)
                fingerprint = hashlib.sha256(prediction.tobytes()).hexdigest()
                final = {'status': 'PASS', 'created_utc': utc_now(), 'model_sha256': sha256(out/'all_dev.joblib'),
                    'training': details, 'all_dev_prediction_sha256': fingerprint,
                    'all_dev_fingerprint_points': len(prediction), 'training_quality_score_computed': False,
                    'interpretation': 'Independent complete-DEV training fingerprint, never a CV or model-selection score.'}
                write_json(final_path, final)
                del model, prediction
                gc.collect()
            record = {'status': 'PASS', 'candidate': candidate, 'created_utc': utc_now(), 'folds': fold_records,
                'all_dev': final, 'fresh_raw_feature_manifest_sha256': sha256(manifest),
                'total_oof_points': sum(row['points'] for row in fold_records), 'seal_rows': 0,
                'all_original_predictions_bitwise_equal': True, 'independent_project_root': str(ROOT),
                'python_executable': sys.executable, 'numerical_versions': versions}
            write_json(out/'RESULTS.json', record)
            candidates_done.append(record)
            del design
            gc.collect()
        del full_design, metadata
        gc.collect()
    write_json(outroot/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(),
        'request_sha256': sha256(request_path), 'workspace_marker_sha256': sha256(ROOT/'INDEPENDENT_NEXT_WORKSPACE.json'),
        'candidates': [row['candidate']['experiment_id'] for row in candidates_done],
        'original_prediction_points_reproduced': sum(row['total_oof_points'] for row in candidates_done),
        'elapsed_seconds': time.perf_counter()-started, 'access_guard': access,
        'source_origin_guard_scope': 'Python audit events block original project file access except canonical raw and the independent virtual environment. Native-library IO is not claimed to be OS sandboxed.',
        'seal_rows': 0, 'reduced_new_uses': 0, 'public_submissions': 0})
    print('INDEPENDENT RAW REPRODUCTION COMPLETE', len(candidates_done), flush=True)


if __name__ == '__main__':
    main()
