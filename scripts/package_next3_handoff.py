"""Snapshot a bounded raw-free NEXT3 payload, then verify its actual Desktop ZIP."""
from pathlib import Path
import argparse
import csv
import datetime
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import zipfile
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, atomic_json, read_json, digest, utc_now
from src.next3.archive_contract import member_name, verify_archive, MAX_UNCOMPRESSED_BYTES
from src.next3.path_budget import check_windows_copy_path_budget
from src.next3.final_source_contract import source_files
from scripts.prepare_next3_independent_reproduction import time_gate

REQUIRED = '''00_GPT_READ_FIRST_NEXT3.md 01_GPT_HANDOFF_NEXT3.md NEXT3_PREFLIGHT.json
BOTTLENECK_BOARD_INITIAL.json BOTTLENECK_BOARD_FINAL.json BOTTLENECK_HISTORY.csv
MAINTENANCE_LOG.json RESOURCE_LOG.csv HANG_RECOVERY_LOG.csv ROUTER_PROTOCOL.md ROUTER_RESULTS.csv
SCALE_DOWN_FORENSIC.md SCALE_REPAIR_RESULTS.csv DEPENDENCE_FORENSIC.md C205_ABLATION_RESULTS.csv
DEPENDENCE_SPECIALIST_RESULTS.csv ORTHOGONAL_COMPLEMENT_RESULTS.csv MODEL_RESULTS_NEXT3.csv CV_RESULTS_NEXT3.csv
PAIRED_BOOTSTRAP_NEXT3.csv ALT_SPLIT_RESULTS_NEXT3.csv REGIME_RESULTS_NEXT3.csv MECHANISM_RESULTS_NEXT3.csv
COUNTEREXAMPLE_RESULTS_NEXT3.csv LONG_NULL_RESULTS_NEXT3.csv LATENCY_NEXT3.json STATE_MEMORY_NEXT3.json
DEV_CANDIDATES_NEXT3.json SEAL_GUARD_NEXT3.json SEAL_OPEN_RECOMMENDATION_NEXT3.md PUBLIC_SUBMISSION_RECOMMENDATION_NEXT3.md
REPRODUCTION_NEXT3.json DETERMINISM_NEXT3.json FINAL_AUDIT_NEXT3.json LIMITATIONS_NEXT3.md NEXT_SESSION_PLAN_NEXT3.md
EXPERIMENT_LEDGER.csv RAW_DATA_REQUIRED_NEXT3.json PORTABLE_RUNTIME_GOLDEN_NEXT3.json DEPLOYMENT_ENTRY_POINTS_NEXT3.json'''.split()
FORBIDDEN_PARTS = {'.git', '.crunchdao', '.venv', '.repro-venv', '.figure-venv', '__pycache__', '.pytest_cache',
    'node_modules', '.next-portable-v2', '.next-portable-v3', 'portable_staging', 'handoff', 'handoff_staging',
    'environment', 'actual_archive_replay'}
REPORT_SUFFIXES = {'.json', '.csv', '.md', '.xml', '.png', '.pdf', '.log'}
RAW_LABELS = {'tau', 'tau_index', 'target', 'break_time', 'break_index', 'label'}
AGGREGATE_ARRAYS = {'AGGREGATE_CONTRAST_DRAWS.npz', 'PAIR_WINDOW_STATISTICS.npz'}


def ordinary_files(directory):
    for parent, directories, filenames in os.walk(directory, followlinks=False):
        parent = Path(parent)
        directories[:] = [name for name in directories if name not in FORBIDDEN_PARTS and
            not (parent/name).is_symlink() and not (parent/name).is_junction()]
        for name in filenames:
            path = parent/name
            if not path.is_symlink() and not path.is_junction(): yield path


def stable_bytes(path):
    for _ in range(3):
        before = path.stat(); content = path.read_bytes(); after = path.stat()
        if before.st_size == after.st_size == len(content) and before.st_mtime_ns == after.st_mtime_ns:
            return content
    raise RuntimeError('Source remained volatile during snapshot: '+str(path))


def raw_record_reason(name, content):
    suffix = Path(name).suffix.lower()
    if suffix == '.csv':
        columns = set(next(csv.reader(io.StringIO(content.decode('utf-8-sig'))), []))
        if columns.intersection({'id', 'dataset_id'}) and columns.intersection(RAW_LABELS):
            return 'Per-real-series target fields'
    if suffix == '.json':
        pending = [json.loads(content)]
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                if isinstance(value.get('historical'), list) and isinstance(value.get('online'), list):
                    return 'Observation-bearing fixture'
                if set(value).intersection({'id', 'dataset_id'}) and set(value).intersection(RAW_LABELS):
                    return 'Per-real-series target record'
                pending.extend(value.values())
            elif isinstance(value, list): pending.extend(value)
    return None


def check_ordinary(path):
    relative = path.relative_to(ROOT).as_posix(); member_name(relative)
    if not path.is_file() or not path.resolve().is_relative_to(ROOT.resolve()):
        raise RuntimeError('Ordinary in-project file required: '+relative)
    for parent in (path, *path.parents):
        if parent == ROOT: break
        if parent.is_symlink() or parent.is_junction(): raise RuntimeError('Linked payload path forbidden: '+relative)
    if path.relative_to(ROOT).parts[0] == 'data' or set(path.relative_to(ROOT).parts)&FORBIDDEN_PARTS:
        raise RuntimeError('Forbidden payload path: '+relative)
    if path.suffix.lower() in ('.zip', '.partial', '.pkl', '.pickle', '.pyc', '.nbi', '.nbc', '.env'):
        raise RuntimeError('Forbidden payload file type: '+relative)
    if any(word in path.name.lower() for word in ('credential', 'access_token', 'refresh_token', 'private_key')):
        raise RuntimeError('Credential-like payload filename: '+relative)
    if path.suffix.lower() == '.parquet' and relative != 'artifacts/validation/GROUP_ASSIGNMENTS.parquet':
        raise RuntimeError('Unexpected data table in payload: '+relative)


def collect_paths(*, require_audit=True):
    missing = [name for name in REQUIRED if not (ROOT/name).is_file()]
    if missing and require_audit: raise RuntimeError('Required NEXT3 reports missing: '+', '.join(missing))
    frozen = read_json(BASE/'preflight/FROZEN_REFERENCES.json')['protected_sha256']
    paths = {ROOT/name for name in frozen}; paths.update(ROOT/name for name in REQUIRED if (ROOT/name).is_file())
    paths.update(ROOT/name for name in ('src/next3/.gitattributes', 'configs/next3/.gitattributes', 'docs/.gitattributes',
        'requirements-lock.txt', 'requirements-next2-figures-lock.txt', 'pyproject.toml'))
    paths.update(ROOT.glob('next3_*.py'))
    paths.update(source_files(ROOT))
    for folder in ('src', 'scripts', 'tests'):
        paths.update(path for path in ordinary_files(ROOT/folder) if path.suffix in ('.py', '.md', '.txt'))
    paths.update(path for path in ordinary_files(ROOT/'configs') if path.suffix == '.json')
    paths.update(path for path in ordinary_files(ROOT/'docs') if path.suffix == '.md' and 'NEXT3' in path.name.upper())
    excluded = {}
    for path in ordinary_files(BASE):
        relative = path.relative_to(ROOT).as_posix()
        if path.name == 'ENVIRONMENT_SETUP.log':
            excluded[relative] = 'Installer log omitted'; continue
        if path.suffix in REPORT_SUFFIXES:
            if path.suffix == '.log' and path.stat().st_size > 2*1024*1024:
                excluded[relative] = 'Large stdout log; terminal receipts retained'; continue
            paths.add(path)
        elif path.suffix == '.py' and ('original_sources' in path.parts or path.name.startswith('original_')):
            paths.add(path)
        elif path.suffix == '.sbnext2' and ('canonical_deployment' in path.parts or 'independent_reproduction' in path.parts):
            paths.add(path)
        elif path.name in AGGREGATE_ARRAYS:
            paths.add(path)
        elif path.suffix == '.npy' and path.name.startswith('fold_') and path.stat().st_size < 20000 and 'forensic' in path.parts:
            paths.add(path)
    paths.discard(ROOT/'SOURCE_MANIFEST_NEXT3.json')
    return sorted(paths), frozen, excluded, missing


def snapshot_payload(destination, *, require_audit=True):
    paths, frozen, excluded, missing = collect_paths(require_audit=require_audit)
    if destination.exists(): raise RuntimeError('Preserve an earlier payload snapshot')
    check_windows_copy_path_budget(destination, [path.relative_to(ROOT).as_posix() for path in paths])
    destination.mkdir(parents=True); files = {}; total = 0
    backup_ledger = (BASE/'preflight/EXPERIMENT_LEDGER_BEFORE_NEXT3.csv').read_bytes()
    for path in paths:
        check_ordinary(path); relative = path.relative_to(ROOT).as_posix(); content = stable_bytes(path)
        sha = hashlib.sha256(content).hexdigest()
        if relative in frozen and sha != frozen[relative]:
            if relative != 'EXPERIMENT_LEDGER.csv' or not content.startswith(backup_ledger):
                raise RuntimeError('Protected old payload changed: '+relative)
        reason = raw_record_reason(relative, content)
        if reason:
            if relative in REQUIRED or relative in frozen: raise RuntimeError('Raw target record in mandatory payload: '+relative)
            excluded[relative] = reason; continue
        if path.suffix in ('.json', '.csv', '.md', '.txt', '.log', '.py'):
            text = content.decode('utf-8-sig')
            if (re.search(r'X-Amz-Signature=[a-fA-F0-9]{32,}', text) or
                    re.search(r'-----BEGIN (?:RSA |EC )?PRIVATE KEY-----\s+[A-Za-z0-9+/]{30}', text)):
                raise RuntimeError('Credential-bearing content in payload: '+relative)
        total += len(content)
        if total >= MAX_UNCOMPRESSED_BYTES-4*1024*1024:
            raise RuntimeError('Payload approaches fixed 256 MiB bound including source manifest')
        target = destination/path.relative_to(ROOT); target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream: stream.write(content)
        files[relative] = {'sha256': sha, 'bytes': len(content)}
    return files, excluded, missing


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--inventory', action='store_true'); args = parser.parse_args()
    if args.inventory:
        paths, frozen, excluded, missing = collect_paths(require_audit=False)
        out = BASE/'handoff/PLANNED_PAYLOAD.json'
        atomic_json(out, {'status': 'PLANNED_NOT_FINAL', 'created_utc': utc_now(),
            'candidate_files': len(paths), 'candidate_bytes_before_content_filter': sum(p.stat().st_size for p in paths),
            'required_reports_missing': missing, 'old_protected_files': len(frozen), 'exclusions': excluded,
            'archive_created': False, 'audit_completed': False})
        print('NEXT3 PLANNED PAYLOAD', len(paths), 'missing final reports', len(missing), flush=True); return
    timing = time_gate()
    audit = read_json(ROOT/'FINAL_AUDIT_NEXT3.json')
    if audit['status'] != 'PASS' or not all(audit['checks'].values()): raise RuntimeError('Complete final audit required')
    for path, sha in audit['evidence_sha256'].items():
        if digest(ROOT/path) != sha: raise RuntimeError('Final audit evidence changed: '+path)
    from src.next.guard import SealGuard
    guard = SealGuard(ROOT); guard.check_state()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    folder = BASE/'handoff'/stamp; folder.mkdir(parents=True, exist_ok=False)
    payload = folder/'payload'; files, excluded, missing = snapshot_payload(payload)
    manifest = {'status': 'FROZEN_NEXT3_PACKAGE_SOURCE_MANIFEST', 'created_utc': utc_now(), 'files': files,
        'files_excluding_this_manifest': len(files), 'uncompressed_bytes_excluding_manifest': sum(item['bytes'] for item in files.values()),
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'final_audit_sha256': digest(ROOT/'FINAL_AUDIT_NEXT3.json'), 'snapshot_clock': timing, 'excluded_reports': excluded,
        'snapshots': 'Each source read is stable across size/mtime checks. Runtime ledgers are frozen copies at packaging time; live originals may continue advancing.',
        'exclusions': ['raw competition observations/labels', 'real point OOF and feature caches', 'credentials/tokens',
            'environments, prior archives, linked workspaces and installer logs'],
        'array_scope': 'Unchanged previously audited bootstrap files and explicitly selected NEXT3 aggregate statistics only',
        'self_hash': 'This manifest and the actual archive are hashed by the external verification receipt; no recursive self hash.'}
    atomic_json(payload/'SOURCE_MANIFEST_NEXT3.json', manifest, immutable=True)
    atomic_json(ROOT/'SOURCE_MANIFEST_NEXT3.json', manifest)
    desktop = Path(subprocess.check_output(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command',
        "[Environment]::GetFolderPath('Desktop')"], text=True).strip()).resolve()
    assert desktop.is_dir()
    destination = desktop/('STRUCTURAL_BREAK_GPT_HANDOFF_NEXT3_'+stamp+'.zip'); temporary = destination.with_suffix('.zip.partial')
    assert not destination.exists() and not temporary.exists() and temporary.parent == destination.parent == desktop
    with temporary.open('xb') as stream:
        with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
            for name in sorted([*files, 'SOURCE_MANIFEST_NEXT3.json']): archive.write(payload/name, name)
        stream.flush(); os.fsync(stream.fileno())
    proof = verify_archive(temporary, expected_manifest_sha256=digest(payload/'SOURCE_MANIFEST_NEXT3.json'))
    temporary.rename(destination)
    receipt = {'status': 'PASS', 'created_utc': utc_now(), 'archive': str(destination), 'bytes': destination.stat().st_size,
        **{k: v for k, v in proof.items() if k != 'files'}, 'snapshot_directory': str(payload),
        'raw_competition_data_included': False, 'credentials_included': False, 'all_seals_unopened': True,
        'reduced_new_usage': 0, 'public_submission_new': 0, 'actual_archive_replay': 'PENDING', 'clock': time_gate()}
    atomic_json(folder/'DESKTOP_ZIP_VERIFICATION_NEXT3.json', receipt, immutable=True)
    atomic_json(BASE/'handoff/DESKTOP_ZIP_VERIFICATION_NEXT3.json', receipt)
    print('NEXT3 DESKTOP ZIP BYTE-VERIFIED; ACTUAL REPLAY STILL REQUIRED', destination, proof['archive_sha256'], flush=True)


if __name__ == '__main__': main()
