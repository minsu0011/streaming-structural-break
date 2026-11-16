"""Every DEV point through the actual packed public API, with write/network denial."""
import os
import sys
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import importlib
import hashlib
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, read_json, atomic_json, digest as file_digest, utc_now, progress
from src.next3.lean_shared_runtime import accounting
from src.next.data import prepare_dev
from scripts.verify_next2_shared_public_api import Datasets


def main():
    data = prepare_dev(ROOT)
    artifact = BASE/'canonical_deployment/PERFORMANCE_lean_shared_v1'
    assembly = read_json(artifact/'ASSEMBLY.json'); assert assembly['status'] == 'PASS'
    for name, sha in assembly['public_source_sha256'].items(): assert file_digest(ROOT/name) == sha, name
    selection = read_json(ROOT/'configs/next3/LEAN_RUNTIME_SELECTION.json')
    assert selection['status'] in ('PROVISIONAL_ENGINEERING_ELIGIBLE', 'VALIDATION_ONLY')
    reference = ROOT/'artifacts/next2/shared_public_api_all_dev/PERFORMANCE/RESULTS.json'
    frozen = read_json(reference); assert frozen['status'] == 'PASS'; expected = frozen['output_fingerprint_sha256']
    out = BASE/'lean_public_api_all_dev'; out.mkdir(parents=True, exist_ok=False)
    identity = {'variant': selection['variant'], 'script_sha256': file_digest(Path(__file__)),
        'model_sha256': file_digest(artifact/'model.sbnext2'), 'descriptor_sha256': file_digest(artifact/'model.json'),
        'assembly_sha256': file_digest(artifact/'ASSEMBLY.json'), 'expected_fingerprint': expected,
        'reference_receipt_sha256': file_digest(reference), 'dev_manifest_sha256': file_digest(data.directory/'MANIFEST.json')}
    atomic_json(out/'POLICY.json', identity, immutable=True)
    active = [True]; attempts = []; network = []; null_opens = []
    def audit(event, arguments):
        if not active[0]: return
        if event in ('socket.connect', 'socket.connect_ex'):
            network.append(event); raise RuntimeError('Public inference attempted network access')
        writing = False
        if event == 'open':
            path, mode, flags = arguments
            writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if writing and isinstance(path, (str, bytes, os.PathLike)) and os.path.normcase(os.path.normpath(os.fsdecode(path))) == os.path.normcase(os.path.normpath(os.devnull)):
                null_opens.append(event); return
        if event in ('os.mkdir', 'os.remove', 'os.rmdir', 'os.rename', 'os.chmod', 'os.truncate', 'os.link', 'os.symlink'):
            writing = True
        if writing:
            attempts.append(event); raise RuntimeError('Public inference attempted persistent filesystem write: '+event)
    sys.addaudithook(audit)
    began = time.perf_counter(); saved = {}; memory = []; modules = {}
    indices = {int(sid): j for j, sid in enumerate(data.ids)}
    sample = read_json(ROOT/'configs/next_latency_policy.json')['series_ids']
    try:
        entry = importlib.import_module('next3_lean_performance_main')
        datasets = Datasets(data, range(len(data.ids))); stream = entry.infer(datasets, artifact)
        assert datasets.accessed == 0 and next(stream) is None and datasets.accessed == 0
        assert set(stream.gi_frame.f_locals) == {'datasets', 'model_directory_path', 'prepared'}
        fingerprint = hashlib.sha256(); points = 0; oo = data.series['online_offsets']
        for j, sid in enumerate(data.ids):
            values = np.empty(int(oo[j+1]-oo[j]), dtype=np.float32)
            for k in range(len(values)):
                value = next(stream)
                assert type(value) is float and np.isfinite(value) and 0 <= value <= 1
                assert datasets.index == j and datasets.current.consumed == k+1
                values[k] = value
            fingerprint.update(values.tobytes()); points += len(values)
            if int(sid) in sample: saved[int(sid)] = values
            if j in (0, len(data.ids)//2, len(data.ids)-1):
                state = stream.gi_frame.f_locals['state']; measured, _, _ = accounting(state)
                memory.append({'series_ordinal': j, **measured})
                del state
            if (j+1) % 1000 == 0: print('NEXT3 LEAN PUBLIC EXACT', j+1, round(time.perf_counter()-began, 1), flush=True)
        try: next(stream)
        except StopIteration: pass
        else: raise AssertionError('Extra output after the complete DEV iterator')
        assert points == len(data.online) and fingerprint.hexdigest() == expected
        reordered = Datasets(data, [indices[sid] for sid in reversed(sample)])
        stream = entry.infer(reordered, artifact); assert next(stream) is None and reordered.accessed == 0
        for sid in reversed(sample):
            actual = np.asarray([next(stream) for _ in range(len(saved[sid]))], dtype=np.float32)
            np.testing.assert_array_equal(actual, saved[sid])
        try: next(stream)
        except StopIteration: pass
        else: raise AssertionError('Extra output after reordered series')
        for name, module in list(sys.modules.items()):
            if name != 'src' and not name.startswith(('src.', 'next3_')): continue
            source = getattr(module, '__file__', None)
            if source is None: continue
            path = Path(source).resolve(); assert path.is_relative_to(ROOT), name
            modules[name] = {'relative': path.relative_to(ROOT).as_posix(), 'sha256': file_digest(path)}
        assert not attempts and not network
    finally:
        active[0] = False
    data.guard.check_state()
    atomic_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(), 'identity': identity,
        'series': len(data.ids), 'points': points, 'output_fingerprint_sha256': fingerprint.hexdigest(),
        'first_none_before_any_dataset_access': True, 'one_current_point_per_output': True,
        'no_length_or_suffix_access': True, 'reverse_order_reset_series': len(sample),
        'persistent_filesystem_write_attempts': 0, 'network_attempts': 0, 'null_device_opens': len(null_opens),
        'decoded_original_model_absent_from_generator_frame': True, 'numeric_memory_probes': memory,
        'project_modules': modules, 'elapsed_seconds': time.perf_counter()-began, 'seal_rows': 0,
        'new_fits': 0, 'in_sample_quality_score_computed': False,
        'scope': 'Actual public API equality and allocation boundaries; final fresh raw reproduction still required.'}, immutable=True)
    progress('Actual packed public API complete', 'All DEV and reversed streams exact; no persistent writes/network; fresh raw reproduction remains', experiment='E003_LEAN_SHARED')
    print('NEXT3 LEAN PUBLIC API PASS', points, flush=True)


if __name__ == '__main__': main()
