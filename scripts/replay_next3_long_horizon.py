"""Optional raw-free PERFORMANCE replay of every frozen long-horizon case hash."""
import os
import sys
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import argparse
import hashlib
import importlib
import importlib.metadata
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next3.artifacts import atomic_json, digest, read_json, utc_now
from src.next3.portable_report_contract import new_output_path, ordinary_payload_file, verify_extracted_manifest
from src.next3.long_horizon_contract import read_golden_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='review_next3_long_horizon.json')
    parser.add_argument('--model-dir', help='Explicit in-copy frozen lean model; default uses final PERFORMANCE delivery descriptor')
    args = parser.parse_args(); began = time.perf_counter()
    proof = verify_extracted_manifest(ROOT); output = new_output_path(ROOT, args.output, proof['files'])
    versions = {}
    for line in (ROOT/'requirements-lock.txt').read_text(encoding='utf-8-sig').splitlines():
        if not line.strip() or line.startswith('#'): continue
        name, version = line.split('=='); actual = importlib.metadata.version(name)
        if actual != version: raise RuntimeError('Complete pinned environment required: '+name)
        versions[name] = actual
    assert len(versions) == 63
    if args.model_dir is None:
        deployment = read_json(ROOT/'DEPLOYMENT_ENTRY_POINTS_NEXT3.json')
        assert deployment['status'] == 'FROZEN_NEXT3_PUBLIC_ENTRY_POINTS'
        specification = deployment['entries']['PERFORMANCE']
        assert specification['entry_module'] == 'next3_lean_performance_main'
        model_name = specification['model_directory']; selection = 'FINAL_PERFORMANCE_DESCRIPTOR'
    else:
        model_name = Path(args.model_dir).as_posix(); selection = 'EXPLICIT_FROZEN_LEAN_MODEL'
    model_file = ordinary_payload_file(ROOT, model_name+'/model.sbnext2')
    descriptor_file = ordinary_payload_file(ROOT, model_name+'/model.json')
    base = ROOT/'artifacts/next3/maintenance'
    protocol_path = base/'LEAN_LONG_HORIZON_PARITY_PROTOCOL.json'
    reference_dir = base/'lean_long_horizon_parity_v1'
    reference = read_json(reference_dir/'RESULTS.json'); protocol = read_json(protocol_path)
    assert reference['status'] == 'PASS' and reference['public_API_prediction_points'] == 34603008
    assert reference['protocol_sha256'] == digest(protocol_path)
    assert digest(model_file) == protocol['canonical_model_sha256']
    for name, sha in protocol['source_sha256'].items(): assert digest(ordinary_payload_file(ROOT, name)) == sha
    for name, sha in reference['files_sha256'].items(): assert digest(reference_dir/name) == sha
    needed = [Path(__file__), model_file, descriptor_file, protocol_path, reference_dir/'RESULTS.json',
              reference_dir/'PER_SYNTHETIC_SERIES.csv', ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json',
              ROOT/'src/next3/long_horizon_contract.py']
    assert all(path.relative_to(ROOT).as_posix() in proof['files'] for path in needed)
    active = [True]; phase = ['INFERENCE']; writes = []; raw_reads = []; network = []
    def audit(event, arguments):
        if not active[0]: return
        if event in ('socket.connect', 'socket.connect_ex'):
            network.append(event); raise RuntimeError('No network during long-horizon replay')
        if event == 'open':
            raw, mode, flags = arguments
            if not isinstance(raw, (str, bytes, os.PathLike)): return
            name = os.fsdecode(raw)
            if os.path.normcase(os.path.normpath(name)) == os.path.normcase(os.path.normpath(os.devnull)): return
            path = Path(name).resolve()
            if path.is_relative_to(ROOT/'data') or (phase[0] == 'INFERENCE' and path.suffix.lower() == '.npy'):
                raw_reads.append(str(path)); raise RuntimeError('Neither competition data nor saved prediction arrays are inputs')
            writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if writing:
                writes.append(str(path)); raise RuntimeError('No persistent writes during long-horizon replay')
        if event in ('os.mkdir', 'os.remove', 'os.rmdir', 'os.rename', 'os.chmod', 'os.truncate', 'os.link', 'os.symlink'):
            writes.append(event); raise RuntimeError('No filesystem mutation during long-horizon replay')
    sys.addaudithook(audit)
    try:
        # Keep the verified entry-first order; the generator has a Numba decorator.
        entry = importlib.import_module('next3_lean_performance_main')
        from scripts.verify_next3_lean_long_horizon_parity import CASES, ScenarioDatasets
        import numpy as np
        suite = read_json(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json')
        assert digest(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json') == protocol['counterexample_lock_sha256']
        expected = read_golden_cases(reference_dir/'PER_SYNTHETIC_SERIES.csv', CASES)
        assert len(expected) == 3072
        completed = 0; points = 0; summaries = []
        for kind, scenario, count, horizon, partition in CASES:
            datasets = ScenarioDatasets(scenario, count, horizon, partition, suite)
            stream = entry.infer(datasets, model_file.parent)
            assert next(stream) is None and datasets.accessed == 0
            fingerprint = hashlib.sha256()
            for row in range(count):
                values = np.empty(horizon, dtype=np.float32)
                for index in range(horizon):
                    value = next(stream)
                    assert type(value) is float and np.isfinite(value) and 0. <= value <= 1.
                    assert datasets.accessed == row+1 and datasets.current.consumed == index+1
                    values[index] = value
                target = expected[(kind, scenario, row)]
                assert datasets.input_sha == target['input_sha256']
                content = values.tobytes()
                assert hashlib.sha256(content).hexdigest() == target['float32_prediction_sha256']
                assert stream.gi_frame.f_locals['state'].state_array_bytes == 16925
                fingerprint.update(content); completed += 1; points += horizon
                if (row+1) % 64 == 0: print('NEXT3 OPTIONAL LONG REPLAY', scenario, row+1, count, points, flush=True)
            sentinel = object(); assert next(stream, sentinel) is sentinel and datasets.accessed == count
            original = [item for item in reference['scenarios'] if item['scenario'] == scenario and item['kind'] == kind]
            assert len(original) == 1 and fingerprint.hexdigest() == original[0]['float32_output_sha256']
            summaries.append({'kind':kind, 'scenario':scenario, 'series':count, 'horizon':horizon,
                              'output_sha256':fingerprint.hexdigest(), 'exact_original_hashes':True})
        assert completed == 3072 and points == 34603008
        modules = {}
        for name, module in list(sys.modules.items()):
            if name != 'src' and not name.startswith(('src.', 'next2_', 'next3_', 'scripts.verify_next3_lean_long_horizon_parity')): continue
            source = getattr(module, '__file__', None)
            if source is None: continue
            path = Path(source).resolve(); assert path.is_relative_to(ROOT), name
            relative = path.relative_to(ROOT).as_posix()
            assert relative in proof['files'] and digest(path) == proof['files'][relative]['sha256']
            modules[name] = {'relative':relative, 'sha256':digest(path)}
        # Other listed historical arrays may be hashed for archive integrity;
        # they are not opened as replay inputs while inference is active.
        phase[0] = 'VERIFY_LISTED_SOURCE_BYTES'
        after = verify_extracted_manifest(ROOT)
        assert after['files'] == proof['files'] and not writes and not raw_reads and not network
    finally:
        active[0] = False
    report = {'status':'PASS', 'created_utc':utc_now(), 'role':'PERFORMANCE', 'entry_module':'next3_lean_performance_main',
        'model_directory':model_name, 'model_selection_authority':selection, 'canonical_model_sha256':digest(model_file),
        'public_API_prediction_points':points, 'synthetic_streams':completed, 'scenarios':summaries,
        'all_input_and_original_float32_output_hashes_exact':True, 'retained_numeric_bytes':16925,
        'source_manifest_sha256':proof['manifest_sha256'], 'files_verified_before_and_after':proof['file_count'],
        'project_modules':modules, 'pinned_versions':versions, 'new_fits':0, 'new_quality_scores':0,
        'saved_prediction_arrays_needed':False, 'raw_competition_inputs_needed':False,
        'persistent_write_attempts':writes, 'competition_or_saved_prediction_access_attempts':raw_reads,
        'network_attempts':network, 'actual_archive_bytes_verified_by_this_command':False,
        'elapsed_seconds_not_latency_benchmark':time.perf_counter()-began, 'script_sha256':digest(Path(__file__)),
        'scope':'Optional complete original long-horizon synthetic hash replay from verified listed source files. Uses generated H/O and compact golden CSV only; no saved point prediction arrays, fits, thresholds or quality metrics. Does not replace actual final archive-byte verification.'}
    atomic_json(output, report, immutable=True)
    print('NEXT3 OPTIONAL LONG-HORIZON REPLAY PASS', points, 'points;', output, flush=True)


if __name__ == '__main__': main()
