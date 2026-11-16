"""Replay only archived source/models under origin-read and inference-write denial."""
from pathlib import Path
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import sys
import time
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'): os.environ[key] = '1'


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--origin', required=True); args = parser.parse_args()
    origin = Path(args.origin).resolve(); payload = Path(__file__).resolve().parents[1]
    if payload.is_relative_to(origin) or origin == payload: raise RuntimeError('Actual independent archive extraction required')
    if not payload.name.startswith('SB_NEXT3_ZIP_REPLAY_'): raise RuntimeError('Explicit actual-archive replay directory required')
    sys.path = [str(payload), *[p for p in sys.path if p and not Path(p).resolve().is_relative_to(origin)]]
    active = [True]; phase = ['SOURCE_VERIFY']; origin_reads = []; writes = []; network = []; null_opens = []
    def audit(event, arguments):
        if not active[0]: return
        if event in ('socket.connect', 'socket.connect_ex'):
            network.append(event); raise RuntimeError('Archived runtime network access forbidden')
        if event == 'open':
            raw, mode, flags = arguments
            if not isinstance(raw, (str, bytes, os.PathLike)): return
            decoded = os.fsdecode(raw)
            if os.path.normcase(os.path.normpath(decoded)) == os.path.normcase(os.path.normpath(os.devnull)):
                null_opens.append(event); return
            path = Path(decoded).resolve()
            writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if path.is_relative_to(origin):
                origin_reads.append(str(path)); raise RuntimeError('Original workspace access forbidden')
            if writing:
                writes.append(str(path)); raise RuntimeError('Persistent write during archived inference forbidden')
        if event in ('os.mkdir', 'os.remove', 'os.rmdir', 'os.rename', 'os.chmod', 'os.truncate', 'os.link', 'os.symlink'):
            writes.append(event); raise RuntimeError('Filesystem mutation during archived inference forbidden')
    sys.addaudithook(audit); began = time.perf_counter()
    try:
        copied = json.loads((payload/'PORTABLE_COPY_MANIFEST_NEXT3.json').read_text(encoding='utf-8'))
        for name, record in copied['files'].items():
            if sha256(payload/name) != record['sha256']: raise RuntimeError('Extracted source mismatch: '+name)
        versions = {}
        for line in (payload/'requirements-lock.txt').read_text(encoding='utf-8').splitlines():
            if not line.strip() or line.startswith('#'): continue
            name, version = line.split('=='); installed = importlib.metadata.version(name)
            if installed != version: raise RuntimeError('Pinned archive replay environment differs: '+name)
            versions[name] = installed
        golden = json.loads((payload/'PORTABLE_RUNTIME_GOLDEN_NEXT3.json').read_text(encoding='utf-8'))
        if golden['status'] != 'FROZEN_SYNTHETIC_PUBLIC_API_GOLDEN': raise RuntimeError('Frozen reference golden required')
        if sha256(payload/'src/next3/portable_probes.py') != golden['probe_source_sha256']:
            raise RuntimeError('Synthetic probe generator changed')
        deployment = json.loads((payload/'DEPLOYMENT_ENTRY_POINTS_NEXT3.json').read_text(encoding='utf-8'))
        if deployment['status'] != 'FROZEN_NEXT3_PUBLIC_ENTRY_POINTS' or set(deployment['entries']) != {'DISTILLED', 'PERFORMANCE'}:
            raise RuntimeError('Exactly two frozen scientific delivery roles required')
        from src.next3.portable_probes import cases, replay
        values = cases(); results = {}; phase[0] = 'INFERENCE'
        allowed = {'DISTILLED': {'next2_canonical_distilled_main'},
            'PERFORMANCE': {'next2_shared_performance_main', 'next3_packed_performance_main', 'next3_lean_performance_main'}}
        for role, specification in deployment['entries'].items():
            module_name = specification['entry_module']
            if module_name not in allowed[role]: raise RuntimeError('Unknown archived public entry')
            directory = (payload/specification['model_directory']).resolve()
            if not directory.is_relative_to(payload): raise RuntimeError('Archived model directory escaped payload')
            if sha256(directory/'model.sbnext2') != golden['roles'][role]['reference_model_sha256']:
                raise RuntimeError('Archived scientific model bytes differ from frozen reference')
            entry = importlib.import_module(module_name); result = replay(entry, directory, values)
            reference = golden['roles'][role]
            for key in ('cases', 'points', 'output_fingerprint_sha256', 'first_none_before_data', 'one_current_point_per_output', 'no_future_iterator_API_access'):
                if result[key] != reference[key]: raise RuntimeError('Archived public inference differs: '+role+'/'+key)
            reverse = replay(entry, directory, list(reversed(values)))
            if {r['case']: r for r in reverse['cases']} != {r['case']: r for r in result['cases']}:
                raise RuntimeError('Archived inference retains state across reordered series')
            results[role] = {**result, 'entry_module': module_name, 'model_directory': specification['model_directory'],
                'model_sha256': sha256(directory/'model.sbnext2'), 'reverse_order_exact': True}
        project_modules = {}
        for name, module in list(sys.modules.items()):
            if name != 'src' and not name.startswith(('src.', 'next2_', 'next3_')): continue
            source = getattr(module, '__file__', None)
            if source is None: continue
            path = Path(source).resolve()
            if not path.is_relative_to(payload): raise RuntimeError('Project module imported outside archive: '+name)
            relative = path.relative_to(payload).as_posix()
            if relative not in copied['files'] or sha256(path) != copied['files'][relative]['sha256']:
                raise RuntimeError('Project module absent from verified archive source manifest: '+name)
            project_modules[name] = {'relative': relative, 'sha256': sha256(path)}
        for name, record in copied['files'].items():
            if sha256(payload/name) != record['sha256']: raise RuntimeError('Archived source/model changed during inference: '+name)
        if origin_reads or writes or network: raise RuntimeError('Archived read-only isolation failed')
    finally: active[0] = False
    from src.next3.artifacts import atomic_json, utc_now
    report = {'status': 'PASS', 'created_utc': utc_now(), 'archive_sha256': copied['archive_sha256'],
        'roles': results, 'public_API_prediction_points': sum(r['points'] for r in results.values())*2,
        'archived_files_verified_before_and_after': len(copied['files']), 'project_modules': project_modules,
        'pinned_versions': versions, 'original_project_access_attempts': origin_reads,
        'persistent_write_attempts': writes, 'network_attempts': network, 'null_device_opens': len(null_opens),
        'raw_competition_observations_used': False, 'new_fits': 0, 'new_quality_scores': 0, 'seal_rows': 0,
        'elapsed_seconds': time.perf_counter()-began, 'script_sha256': sha256(Path(__file__)),
        'boundary': 'Actual archive extraction and archived modules in a pinned sibling environment; Python audit hook is not an OS sandbox.'}
    atomic_json(payload/'ARCHIVED_RUNTIME_RESULT_NEXT3.json', report, immutable=True)
    print('NEXT3 ACTUAL ARCHIVED RUNTIME PASS', report['public_API_prediction_points'], flush=True)


if __name__ == '__main__': main()
