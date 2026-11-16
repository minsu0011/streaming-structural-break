"""NEXT3 final-audit child: raw public training, isolated source and expected weights."""
import os
import sys
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import argparse
import json
import hashlib
import importlib
import importlib.metadata
import time


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--origin', required=True)
    parser.add_argument('--role', choices=['DISTILLED', 'PERFORMANCE'], required=True); args = parser.parse_args()
    origin = Path(args.origin).resolve(); payload = Path(__file__).resolve().parents[1]; workspace = payload.parent
    if payload.name != 'payload' or workspace.is_relative_to(origin) or Path(sys.prefix).resolve() != workspace/'environment':
        raise RuntimeError('Fresh sibling payload and its own environment required')
    out = payload/'artifacts/independent'/args.role
    if out.exists(): raise RuntimeError('Never overwrite an independent training run')
    out.mkdir(parents=True)
    temporary = workspace/'temporary'/args.role; temporary.mkdir(parents=True)
    os.environ['TMP'] = os.environ['TEMP'] = str(temporary)
    sys.path = [str(payload), *[p for p in sys.path if p and not Path(p).resolve().is_relative_to(origin)]]
    denied = []; expected_reads_before_training = []; outside_writes = []; network = []
    phase = ['TRAINING']; reads = [0]
    def audit(event, arguments):
        if event in ('socket.connect', 'socket.connect_ex'):
            network.append(event); raise RuntimeError('Independent reproduction network access denied')
        if event == 'open':
            raw, mode, flags = arguments
            if not isinstance(raw, (str, bytes, os.PathLike)): return
            decoded = os.fsdecode(raw)
            if os.path.normcase(os.path.normpath(decoded)) == os.path.normcase(os.path.normpath(os.devnull)): return
            path = Path(decoded).resolve()
            writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if path.is_relative_to(origin):
                denied.append(str(path)); raise RuntimeError('Original project access denied')
            if writing and not path.is_relative_to(workspace):
                outside_writes.append(str(path)); raise RuntimeError('Write outside reproduction workspace denied')
            if not writing and phase[0] == 'TRAINING' and path.is_relative_to(payload/'expected'):
                expected_reads_before_training.append(str(path)); raise RuntimeError('Expected weights inaccessible until fitting completes')
            if not writing: reads[0] += 1
        if event in ('os.remove', 'os.rmdir', 'os.mkdir', 'os.chmod', 'os.truncate'):
            raw = arguments[0]
            if isinstance(raw, (str, bytes, os.PathLike)) and not Path(os.fsdecode(raw)).resolve().is_relative_to(workspace):
                outside_writes.append(event); raise RuntimeError('Mutation outside reproduction workspace denied')
        if event in ('os.rename', 'os.link', 'os.symlink'):
            for raw in arguments[:2]:
                if isinstance(raw, (str, bytes, os.PathLike)) and not Path(os.fsdecode(raw)).resolve().is_relative_to(workspace):
                    outside_writes.append(event); raise RuntimeError('Mutation outside reproduction workspace denied')
    sys.addaudithook(audit)
    manifest = json.loads((payload/'COPIED_SOURCE_MANIFEST.json').read_text())
    for relative, digest in manifest['files_sha256'].items():
        assert digest_file(payload/relative) == digest, relative
    versions = {}
    for line in (payload/'requirements-lock.txt').read_text().splitlines():
        if not line.strip() or line.startswith('#'): continue
        name, version = line.split('=='); installed = importlib.metadata.version(name)
        assert installed == version, (name, installed, version); versions[name] = installed
    entry_plan = json.loads((payload/'REPRODUCTION_ENTRY_LOCK.json').read_text(encoding='utf-8'))
    entry_name = entry_plan['entries'][args.role]
    allowed = {'DISTILLED': ['next2_canonical_distilled_main'], 'PERFORMANCE': ['next2_shared_performance_main', 'next3_packed_performance_main', 'next3_lean_performance_main']}
    if entry_name not in allowed[args.role]: raise RuntimeError('Unregistered reproduction entry')
    entry = importlib.import_module(entry_name)
    import numpy as np
    from src.next.guard import SealGuard
    from src.next.data import iter_dev_series, RAW_SHA
    from src.next2.deployment_training import OpaqueSeal
    from src.next2.canonical_model import dumps, load
    from src.utils.artifacts import write_json, utc_now
    guard = SealGuard(payload)
    raw_before = {name: digest_file(payload/'data/raw'/name) for name in RAW_SHA}
    assert raw_before == RAW_SHA
    training_seen = [0]
    def raw_training():
        for row in iter_dev_series(guard):
            training_seen[0] += 1; yield row
        for sid in sorted(guard.seal_ids):
            yield sid, OpaqueSeal(), OpaqueSeal(), OpaqueSeal()
    began = time.perf_counter(); model_dir = out/'model'
    descriptor = entry.train(raw_training(), model_dir)
    trained = time.perf_counter(); assert training_seen[0] == 8000
    phase[0] = 'VERIFY_AFTER_TRAINING'
    expected = json.loads((payload/'expected/EXPECTED.json').read_text())[args.role]
    assert digest_file(model_dir/'model.sbnext2') == expected['canonical_model_sha256']
    assert (model_dir/'model.sbnext2').read_bytes() == (payload/expected['model_relative']).read_bytes()
    restored = load(model_dir/'model.sbnext2'); assert dumps(restored) == (model_dir/'model.sbnext2').read_bytes()
    del restored
    training = descriptor['training']; components = training['component_receipts']
    assert set(components) == set(expected['component_training'])
    comparisons = {}
    for role, reference in expected['component_training'].items():
        receipt = components[role]
        assert receipt['status'] == 'PASS' and receipt['seal_rows_fitted'] == 0
        assert receipt['seal_series_skipped_before_value_or_label_access'] == 2000
        checks = {key: receipt['training'][key] == reference[key] for key in
            ('feature_matrix_sha256', 'ordered_ids_sha256', 'ordered_targets_sha256', 'training_rows', 'training_series')}
        assert all(checks.values()); comparisons[role] = checks
    assert training['new_fits'] == (1 if args.role == 'DISTILLED' else 2)
    assert training['seal_series_skipped_before_any_value_access'] == 2000
    class Online:
        def __init__(self, values): self.values = values; self.consumed = 0; self.iterated = False
        def __iter__(self):
            if self.iterated: raise AssertionError('Online consumed twice')
            self.iterated = True; return self
        def __next__(self):
            if self.consumed == len(self.values): raise StopIteration
            value = self.values[self.consumed]; self.consumed += 1; return value
        def __len__(self): raise AssertionError('Future length inspected')
        def __length_hint__(self): raise AssertionError('Future length hint inspected')
        def __array__(self, *args, **kwargs): raise AssertionError('Future suffix materialized')
    class Datasets:
        def __init__(self): self.rows = iter(iter_dev_series(guard)); self.current = None; self.sid = None; self.accessed = 0
        def __iter__(self): return self
        def __next__(self):
            if self.current is not None: assert self.current.consumed == len(self.current.values)
            sid, historical, online, tau = next(self.rows)
            self.sid = sid; self.current = Online(online); self.accessed += 1
            return historical, self.current
        def __len__(self): raise AssertionError('Dataset length inspected')
        def __length_hint__(self): raise AssertionError('Dataset length hint inspected')
    datasets = Datasets(); stream = entry.infer(datasets, model_dir)
    assert next(stream) is None and datasets.accessed == 0
    prediction_hash = hashlib.sha256(); seen = []; points = 0; previous_sid = None; values = []
    for prediction in stream:
        assert type(prediction) is float and np.isfinite(prediction) and 0 <= prediction <= 1
        if datasets.sid != previous_sid:
            if previous_sid is not None: prediction_hash.update(np.asarray(values, dtype=np.float32).tobytes())
            previous_sid = datasets.sid; seen.append(previous_sid); values = []
        assert datasets.current.consumed == len(values)+1
        values.append(prediction); points += 1
    prediction_hash.update(np.asarray(values, dtype=np.float32).tobytes())
    assert len(seen) == len(set(seen)) == expected['series'] and set(seen) == set(guard.dev_ids)
    assert points == expected['points'] and prediction_hash.hexdigest() == expected['all_dev_prediction_sha256']
    raw_after = {name: digest_file(payload/'data/raw'/name) for name in RAW_SHA}; assert raw_after == raw_before
    guard.check_state()
    for relative, expected_digest in manifest['files_sha256'].items():
        assert digest_file(payload/relative) == expected_digest, relative
    modules = {}
    for name, module in list(sys.modules.items()):
        if name != 'src' and not name.startswith(('src.', 'next2_', 'next3_')): continue
        source = getattr(module, '__file__', None)
        if source is None: continue
        path = Path(source).resolve(); assert path.is_relative_to(payload), name
        modules[name] = {'relative': path.relative_to(payload).as_posix(), 'sha256': digest_file(path)}
    assert modules and not denied and not expected_reads_before_training and not outside_writes and not network
    report = {'status': 'PASS', 'created_utc': utc_now(), 'role': args.role, 'entry_module': entry_name, 'runtime_variant': descriptor.get('variant', descriptor.get('runtime')), 'source_workspace': str(workspace),
        'canonical_model_sha256': digest_file(model_dir/'model.sbnext2'), 'canonical_model_bytes_exact': True,
        'all_dev_prediction_sha256': prediction_hash.hexdigest(), 'series': len(seen), 'points': points,
        'training_comparisons': comparisons, 'new_fits': training['new_fits'], 'seal_rows_fitted': 0,
        'seal_values_accessed': 0, 'opaque_seal_series_skipped': 2000, 'raw_before_sha256': raw_before, 'raw_after_sha256': raw_after,
        'first_none_before_data': True, 'one_current_point_per_yield': True, 'no_future_length_or_suffix_access': True,
        'original_project_access_attempts': denied, 'expected_reads_before_training': expected_reads_before_training,
        'outside_workspace_write_attempts': outside_writes, 'network_attempts': network, 'guarded_read_opens': reads[0],
        'project_modules': modules, 'pinned_versions': versions, 'pin_count': len(versions),
        'training_seconds': trained-began, 'total_seconds': time.perf_counter()-began,
        'training_quality_score_computed': False, 'source_manifest_sha256': digest_file(payload/'COPIED_SOURCE_MANIFEST.json'),
        'script_sha256': digest_file(Path(__file__)),
        'scope': 'Actual canonical public train and infer in a fresh pinned environment, from ordinary raw-byte copies and no previous feature caches. Python audit hook is not an OS sandbox.'}
    write_json(out/'RESULTS.json', report)
    print('NEXT3 INDEPENDENT TRAINING PASS', args.role, points, flush=True)


if __name__ == '__main__': main()
