"""Register the new session and restore previous evidence without fitting."""
from pathlib import Path
import sys
import time
import os
import json
import subprocess
import importlib.metadata
import shutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, utc_now, digest, read_json, atomic_json, progress
from src.next.guard import SealGuard, SealAccessError
from src.next.data import RAW_SHA


def main():
    if (BASE/'SESSION.json').exists():
        raise RuntimeError('NEXT3 already initialized; use resume instead')
    session = {'status': 'PREFLIGHT_RUNNING', 'started_utc': utc_now(),
        'monotonic_start': time.monotonic(), 'pid_at_start': os.getpid(),
        'target_research_seconds': 40200, 'target_audit_seconds': 3000,
        'target_total_seconds': 43200, 'credited_productive_seconds': 0,
        'lost_or_unobserved_seconds': 0, 'segments': [],
        'timing_rule': 'Monotonic wall time; productive intervals credited only with recent agent activity or active child evidence. Offline gaps excluded.',
        'all_seals_unopened': True, 'reduced_new_usage': 0,
        'public_submission_new': 0, 'crunch_cloud_submission_new': 0}
    atomic_json(BASE/'SESSION.json', session, immutable=True)
    progress('Initialize NEXT3', 'Restore hashes, existing selection, guarded metadata and environment')
    source = Path(r'C:\Users\minsu\.codex\attachments\4eb83de0-fb35-4fe1-b39d-e16277bfa545\pasted-text.txt')
    target = ROOT/'docs/USER_SPECIFICATION_NEXT3.md'
    with target.open('xb') as stream:
        stream.write(source.read_bytes())
    status = subprocess.check_output(['git', 'status', '--porcelain=v1', '-uall'], cwd=ROOT)
    (BASE/'preflight').mkdir(parents=True, exist_ok=True)
    (BASE/'preflight/INITIAL_GIT_STATUS.txt').write_bytes(status)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    guard = SealGuard(ROOT)
    guard.admit(purpose='NEXT3 training/validation/OOF/candidate preflight')
    blocked = {}
    for purpose in ('training', 'validation', 'OOF', 'candidate generation', 'label access'):
        try:
            guard.admit([min(guard.seal_ids)], purpose=purpose)
        except SealAccessError:
            blocked[purpose] = True
        else:
            raise RuntimeError('Seal admission unexpectedly allowed')
    raw = {name: digest(ROOT/'data/raw'/name) for name in RAW_SHA}
    if raw != RAW_SHA:
        raise RuntimeError('Raw hashes changed: modeling prohibited')
    manifest = read_json(ROOT/'SOURCE_MANIFEST_NEXT2.json')
    changed = []
    protected = {}
    for name, entry in manifest['files'].items():
        actual = digest(ROOT/name)
        if actual != entry['sha256']:
            changed.append(name)
        protected[name] = actual
    if changed:
        atomic_json(BASE/'preflight/FROZEN_FILE_FAILURE.json', {'status': 'FAIL', 'changed': changed})
        raise RuntimeError('Previous handoff sources/artifacts changed; inspect before modeling')
    protected['SOURCE_MANIFEST_NEXT2.json'] = digest(ROOT/'SOURCE_MANIFEST_NEXT2.json')
    # The cumulative ledger can later be extended; its starting bytes remain here.
    shutil.copyfile(ROOT/'EXPERIMENT_LEDGER.csv', BASE/'preflight/EXPERIMENT_LEDGER_BEFORE_NEXT3.csv')
    selection = read_json(ROOT/'DEV_CANDIDATES_NEXT2.json')
    roles = {name: selection[key] for name, key in (
        ('FROZEN_DISTILLED', 'distilled'), ('FROZEN_PERFORMANCE', 'performance'))}
    scientific = ROOT/'artifacts/next/refits/J102_SERIAL_VARIANCE_512/cache_research_v1/model.joblib'
    roles['FROZEN_J102_SCIENTIFIC'] = {'model': scientific.relative_to(ROOT).as_posix(), 'sha256': digest(scientific)}
    atomic_json(BASE/'preflight/FROZEN_REFERENCES.json', {
        'created_utc': utc_now(), 'roles': roles, 'protected_sha256': protected,
        'allowed_append_only_exception': 'EXPERIMENT_LEDGER.csv; every previous row must remain semantically exact',
        'raw_sha256': raw, 'split_sha256': guard.split.digest,
        'seal_lock_sha256': guard.lock_sha, 'reduced_ledger_sha256': guard.ledger_sha,
        'initial_commit': commit, 'user_spec_sha256': digest(target)}, immutable=True)
    required = [line.strip() for line in (ROOT/'requirements-lock.txt').read_text().splitlines()
                if line.strip() and not line.startswith('#')]
    versions = {}
    mismatches = []
    for line in required:
        name, expected = line.split('==', 1)
        actual = importlib.metadata.version(name)
        versions[name] = actual
        if actual != expected:
            mismatches.append({'package': name, 'expected': expected, 'actual': actual})
    if mismatches:
        raise RuntimeError('Pinned environment mismatch: ' + json.dumps(mismatches))
    report = {'status': 'PARTIAL_PREFLIGHT_HASHES_AND_GUARD_PASS', 'created_utc': utc_now(),
        'guard_checks': blocked, 'DEV_series': len(guard.dev_ids), 'seal_series': len(guard.seal_ids),
        'DEV_seal_overlap': len(guard.dev_ids & guard.seal_ids), 'raw_sha256': raw,
        'protected_files_exact': len(protected), 'pinned_packages': versions,
        'python': sys.version, 'initial_git_commit': commit,
        'git_status_sha256': digest(BASE/'preflight/INITIAL_GIT_STATUS.txt'),
        'pending': ['Fresh scorer/guard tests', 'Fresh frozen public API prediction fingerprints'],
        'all_seals_unopened': True, 'reduced_new_usage': 0, 'public_submission_new': 0,
        'crunch_cloud_submission_new': 0}
    atomic_json(BASE/'preflight/INITIAL_RESTORE.json', report, immutable=True)
    progress('NEXT3 restored', 'All previous manifest files exact; raw/seal guard and pinned dependencies pass')
    print('NEXT3 INITIAL RESTORE PASS', len(protected), 'protected files;', len(versions), 'exact package pins', flush=True)


if __name__ == '__main__':
    main()
