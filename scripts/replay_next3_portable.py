"""Reviewer command: validate extracted inputs and replay two raw-free model roles."""
from pathlib import Path
import argparse
import importlib
import importlib.metadata
import os
import sys
import time
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
from src.next3.artifacts import atomic_json, digest, read_json, utc_now
from src.next3.portable_report_contract import new_output_path, ordinary_payload_file, verify_extracted_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='review_next3_replay.json', help='New JSON file; existing paths are preserved')
    args = parser.parse_args(); began = time.perf_counter()
    proof = verify_extracted_manifest(ROOT); output = new_output_path(ROOT, args.output, proof['files'])
    versions = {}
    for line in (ROOT/'requirements-lock.txt').read_text(encoding='utf-8-sig').splitlines():
        if not line.strip() or line.startswith('#'): continue
        name, version = line.split('=='); installed = importlib.metadata.version(name)
        if installed != version: raise RuntimeError('Review environment pin differs: '+name)
        versions[name] = installed
    if len(versions) != 63: raise RuntimeError('The complete 63-pin review environment is required')
    golden = read_json(ROOT/'PORTABLE_RUNTIME_GOLDEN_NEXT3.json')
    deployment = read_json(ROOT/'DEPLOYMENT_ENTRY_POINTS_NEXT3.json')
    if golden['status'] != 'FROZEN_SYNTHETIC_PUBLIC_API_GOLDEN': raise RuntimeError('Frozen synthetic probe result required')
    if deployment['status'] != 'FROZEN_NEXT3_PUBLIC_ENTRY_POINTS' or set(deployment['entries']) != {'DISTILLED', 'PERFORMANCE'}:
        raise RuntimeError('Final two-role delivery descriptor required')
    if digest(ROOT/'src/next3/portable_probes.py') != golden['probe_source_sha256']:
        raise RuntimeError('Synthetic probe source differs from frozen reference')
    for name in ('requirements-lock.txt', 'PORTABLE_RUNTIME_GOLDEN_NEXT3.json', 'DEPLOYMENT_ENTRY_POINTS_NEXT3.json',
                 'src/next3/portable_probes.py', Path(__file__).relative_to(ROOT).as_posix()):
        if name not in proof['files']: raise RuntimeError('Required replay source absent from source manifest: '+name)
    active = [True]; writes = []; network = []; raw_reads = []
    def audit(event, arguments):
        if not active[0]: return
        if event in ('socket.connect', 'socket.connect_ex'):
            network.append(event); raise RuntimeError('Network access during portable inference is forbidden')
        if event == 'open':
            raw, mode, flags = arguments
            if not isinstance(raw, (str, bytes, os.PathLike)): return
            name = os.fsdecode(raw)
            if os.path.normcase(os.path.normpath(name)) == os.path.normcase(os.path.normpath(os.devnull)): return
            path = Path(name).resolve()
            if path.is_relative_to(ROOT/'data'):
                raw_reads.append(str(path)); raise RuntimeError('Competition data is not an input to this replay')
            writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if writing:
                writes.append(str(path)); raise RuntimeError('Persistent write during portable inference is forbidden')
        if event in ('os.mkdir', 'os.remove', 'os.rmdir', 'os.rename', 'os.chmod', 'os.truncate', 'os.link', 'os.symlink'):
            writes.append(event); raise RuntimeError('Filesystem mutation during portable inference is forbidden')
    sys.addaudithook(audit)
    try:
        from src.next3.portable_probes import cases, replay
        values = cases(); results = {}
        allowed = {'DISTILLED': {'next2_canonical_distilled_main'},
            'PERFORMANCE': {'next2_shared_performance_main', 'next3_packed_performance_main', 'next3_lean_performance_main'}}
        for role in ('DISTILLED', 'PERFORMANCE'):
            specification = deployment['entries'][role]; module_name = specification['entry_module']
            if module_name not in allowed[role]: raise RuntimeError('Unknown public model role')
            model_name = specification['model_directory']+'/model.sbnext2'
            if model_name not in proof['files']: raise RuntimeError('Model is absent from extracted manifest')
            model_path = ordinary_payload_file(ROOT, model_name)
            if digest(model_path) != golden['roles'][role]['reference_model_sha256']:
                raise RuntimeError('Scientific model differs from frozen reference')
            if digest(model_path) != specification['canonical_model_sha256']:
                raise RuntimeError('Delivery model checksum differs')
            entry = importlib.import_module(module_name)
            result = replay(entry, model_path.parent, values); expected = golden['roles'][role]
            for key in ('cases', 'points', 'output_fingerprint_sha256', 'first_none_before_data',
                        'one_current_point_per_output', 'no_future_iterator_API_access'):
                if result[key] != expected[key]: raise RuntimeError('Portable inference differs: '+role+'/'+key)
            reverse = replay(entry, model_path.parent, list(reversed(values)))
            if {row['case']: row for row in reverse['cases']} != {row['case']: row for row in result['cases']}:
                raise RuntimeError('Portable model does not reset between series')
            results[role] = {**result, 'reverse_order_exact': True, 'entry_module': module_name,
                'model_directory': specification['model_directory'], 'canonical_model_sha256': digest(model_path)}
        modules = {}
        for name, module in list(sys.modules.items()):
            if name != 'src' and not name.startswith(('src.', 'next2_', 'next3_')): continue
            filename = getattr(module, '__file__', None)
            if filename is None: continue
            source = Path(filename).resolve()
            if not source.is_relative_to(ROOT): raise RuntimeError('Project module loaded outside extracted directory: '+name)
            relative = source.relative_to(ROOT).as_posix()
            if relative not in proof['files'] or digest(source) != proof['files'][relative]['sha256']:
                raise RuntimeError('Unverified project module: '+name)
            modules[name] = {'file': relative, 'sha256': digest(source)}
        after = verify_extracted_manifest(ROOT)
        if after['files'] != proof['files']: raise RuntimeError('Extracted inputs changed during inference')
        if writes or raw_reads or network: raise RuntimeError('Portable replay audit reported forbidden access')
    finally: active[0] = False
    report = {'status': 'PASS', 'created_utc': utc_now(), 'roles': results,
        'public_API_prediction_points': sum(record['points'] for record in results.values())*2,
        'source_manifest_sha256': proof['manifest_sha256'], 'files_verified_before_and_after': proof['file_count'],
        'pinned_versions': versions, 'project_modules': modules, 'persistent_write_attempts_during_inference': writes,
        'network_attempts': network, 'competition_data_access_attempts': raw_reads,
        'new_fits': 0, 'new_quality_scores': 0, 'raw_competition_inputs_needed': False,
        'actual_archive_bytes_independently_verified_by_this_command': False,
        'scope': 'Reviewer-owned extracted source replay. Actual archive SHA/CRC and origin-denied independent extraction are separately reported by the final archive auditor.',
        'elapsed_seconds': time.perf_counter()-began, 'source_sha256': digest(Path(__file__))}
    atomic_json(output, report, immutable=True)
    print('NEXT3 PORTABLE REPLAY PASS', report['public_API_prediction_points'], 'points;', output, flush=True)


if __name__ == '__main__': main()
