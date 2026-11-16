"""Explicit raw-free payload and duration-gated, fully verified Desktop ZIP."""
from pathlib import Path
import sys
import os
import json
import csv
import hashlib
import subprocess
import zipfile
import argparse
import datetime
import re
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.next2.session import require_handoff_time
from src.next.guard import SealGuard
from src.utils.artifacts import write_json, sha256, utc_now

REQUIRED = '''00_GPT_READ_FIRST_NEXT2.md 01_GPT_HANDOFF_NEXT2.md CURRENT_GENERATION_COMPONENT_GRAPH.md
DISTILLATION_RESULTS.csv RUNTIME_PRUNING_RESULTS.csv SCALE_DOWN_RESULTS.csv HEAVY_TAIL_ROBUSTNESS_RESULTS.csv
ARCH_ROBUSTNESS_RESULTS.csv LONG_NULL_RESULTS_NEXT2.csv EVIDENCE_MEMORY_RESULTS.csv BAYES_AGE_ABLATION.csv
GLR_NECESSITY_RESULTS.csv SERIAL_CORRECTION_RESULTS.csv COMPLEMENT_STANDALONE_RESULTS.csv COMPLEMENT_CORRELATION.csv
COMPLEMENT_BLEND_RESULTS.csv COUNTEREXAMPLE_SUITE_LOCK.json COUNTEREXAMPLE_RESULTS.csv MODEL_RESULTS_NEXT2.csv
CV_RESULTS_NEXT2.csv PAIRED_BOOTSTRAP_NEXT2.csv ALT_SPLIT_RESULTS_NEXT2.csv REGIME_RESULTS_NEXT2.csv
MECHANISM_RESULTS_NEXT2.csv LATENCY_NEXT2.json STATE_MEMORY_NEXT2.json DEV_CANDIDATES_NEXT2.json
SEAL_GUARD_NEXT2.json SEAL_OPEN_RECOMMENDATION_NEXT2.md REPRODUCTION_NEXT2.json DETERMINISM_NEXT2.json
FINAL_AUDIT_NEXT2.json LIMITATIONS_NEXT2.md NEXT_SESSION_PLAN_NEXT2.md EXPERIMENT_LEDGER.csv
RAW_DATA_REQUIRED_NEXT2.json SEAL_PARTITION_500x4_LOCK.json REDUCED_TEST_USAGE_LEDGER.json ROBUSTNESS_SPLIT_LOCK.json'''.split()
FORBIDDEN = {'.git', '.crunchdao', '.venv', '.repro-venv', '.figure-venv', '__pycache__',
    'handoff', 'portable_staging', 'node_modules', '.next-portable-v2', '.next-portable-v3'}
TEXT_REPORTS = {'.json', '.csv', '.md', '.xml', '.png', '.pdf'}
RAW_LABELS = {'tau', 'tau_index', 'target', 'break_time', 'break_index', 'label'}


def ordinary_files(directory):
    for parent, dirs, files in os.walk(directory, followlinks=False):
        parent = Path(parent)
        dirs[:] = [n for n in dirs if n not in FORBIDDEN and not (parent/n).is_symlink() and not (parent/n).is_junction()]
        for name in files:
            path = parent/name
            if not path.is_symlink() and not path.is_junction(): yield path


def raw_record_reason(path):
    if path.suffix == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as stream: columns = set(next(csv.reader(stream), []))
        if columns.intersection({'id', 'dataset_id'}) and columns.intersection(RAW_LABELS): return 'Per-real-series target fields'
    if path.suffix == '.json':
        pending = [json.loads(path.read_text(encoding='utf-8-sig'))]
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                if isinstance(value.get('historical'), list) and isinstance(value.get('online'), list): return 'Observation-bearing fixture'
                if set(value).intersection({'id', 'dataset_id'}) and set(value).intersection(RAW_LABELS): return 'Per-real-series target record'
                pending.extend(value.values())
            elif isinstance(value, list): pending.extend(value)
    return None


def collect_payload(*, require_audit=True):
    required = REQUIRED if require_audit else [p for p in REQUIRED if p != 'FINAL_AUDIT_NEXT2.json']
    missing = [p for p in required if not (ROOT/p).is_file()]
    if missing: raise RuntimeError('Required reports missing: '+', '.join(missing))
    selection = json.loads((ROOT/'DEV_CANDIDATES_NEXT2.json').read_text()); assert selection['status'] == 'FROZEN_NEXT2_DEV_SELECTION'
    paths = {ROOT/p for p in required}; excluded = {}
    paths.update(ROOT/('SEAL_'+letter+'_IDS.txt') for letter in 'ABCD')
    paths.update(ROOT/p for p in ('requirements-lock.txt', 'requirements-next2-figures-lock.txt', '.gitattributes', 'artifacts/validation/SPLIT_LOCK_REAL.json', 'artifacts/validation/GROUP_ASSIGNMENTS.parquet'))
    paths.update(ROOT.glob('next2_*.py'))
    for folder in ('src', 'scripts', 'tests', 'vendor/official'):
        paths.update(p for p in ordinary_files(ROOT/folder) if p.suffix in {'.py', '.md', '.txt'})
    paths.update(p for p in ordinary_files(ROOT/'configs') if p.suffix == '.json')
    paths.update(p for p in ordinary_files(ROOT/'docs') if p.suffix == '.md' and ('NEXT' in p.name.upper() or 'STRUCTURAL' in p.name.upper()))
    paths.update((ROOT/'checkpoints').glob('NEXT2_*.json'))
    for path in ordinary_files(ROOT/'artifacts/next2'):
        if path.suffix in TEXT_REPORTS:
            if path.name in ('COPIED_SOURCE_MANIFEST_BEFORE_CONFIG_COMPLETION.json',): continue
            if path.name == 'SNAPSHOTS.csv' and 'synthetic_attribution' in path.parts:
                excluded[path.relative_to(ROOT).as_posix()] = 'Detailed synthetic per-seed SHAP table omitted; grouped summaries and provenance retained'
                continue
            reason = raw_record_reason(path)
            if reason: excluded[path.relative_to(ROOT).as_posix()] = reason
            else: paths.add(path)
    # Original J102 paired score draws are the fixed comparator for the new best.
    for eid, stage in [('J102_SERIAL_VARIANCE_512', 'next'), ('BL2_C205_DEPENDENCE_80', 'next2')]:
        directory = ROOT/'artifacts'/stage/'forensic'/eid
        paths.update(p for p in directory.glob('*') if p.suffix in {'.json', '.csv'} or (p.suffix == '.npy' and p.name.startswith('fold_') and p.stat().st_size < 20000))
    binary_models = set()
    # All canonical versions/references are small and document exact rejected/selected runtimes.
    binary_models.update((ROOT/'artifacts/next2/canonical_deployment').rglob('*.sbnext2'))
    binary_models.update((ROOT/'artifacts/next2/independent_reproduction').rglob('*.sbnext2'))
    for reference in selection['frozen_references'].values():
        model = ROOT/reference['model_path']; assert sha256(model) == reference['model_sha256']
        binary_models.add(model)
        if (model.parent/'model.json').exists(): paths.add(model.parent/'model.json')
    c_model = ROOT/'artifacts/next2/refits/C205_DEPENDENCE/cache_v1/model.joblib'
    assert sha256(c_model) == json.loads((c_model.parent/'TRAINING.json').read_text())['model_sha256']
    binary_models.add(c_model)
    paths.update(binary_models)
    # These arrays contain resampled aggregate statistics, never observations or real point OOF.
    paths.add(ROOT/'artifacts/next2/channel_response/BOOTSTRAP_VECTORS.npy')
    for folder in ('alarm_uncertainty', 'counterexample_bootstrap', 'cross_background_bootstrap'):
        paths.update(p for p in (ROOT/'artifacts/next2'/folder).rglob('*.npz') if p.name in ('BOOTSTRAP_VECTORS.npz', 'THRESHOLDS.npz', 'TRAJECTORY_MAXIMA.npz'))
    for role in ('DISTILLED', 'PERFORMANCE'):
        path = ROOT/'artifacts/next2/independent_reproduction'/role/'RUN.log'
        if path.exists(): paths.add(path)
    for path in sorted(paths):
        relative = path.relative_to(ROOT)
        if not path.is_file() or path.is_symlink() or path.is_junction(): raise RuntimeError('Ordinary payload file required: '+str(relative))
        if relative.parts[0] == 'data' or any(p in FORBIDDEN for p in relative.parts): raise RuntimeError('Forbidden path: '+str(relative))
        if path.suffix.lower() in ('.zip', '.pkl', '.pickle', '.pyc', '.nbi', '.nbc', '.env'): raise RuntimeError('Forbidden file type')
        if any(word in path.name.lower() for word in ('credential', 'access_token', 'refresh_token', 'private_key')): raise RuntimeError('Credential-like filename')
        if path.suffix == '.parquet' and relative.as_posix() != 'artifacts/validation/GROUP_ASSIGNMENTS.parquet': raise RuntimeError('Unexpected parquet')
        if path.suffix in ('.joblib', '.sbnext2') and path not in binary_models: raise RuntimeError('Unexpected model')
        reason = raw_record_reason(path)
        if reason: raise RuntimeError('Raw record in required/source file: '+str(relative))
        if path.suffix in {'.json', '.csv', '.md', '.txt', '.log', '.py'}:
            text = path.read_text(encoding='utf-8-sig')
            if re.search(r'X-Amz-Signature=[a-fA-F0-9]{32,}', text) or re.search(r'-----BEGIN (?:RSA |EC )?PRIVATE KEY-----\s+[A-Za-z0-9+/]{30}', text):
                raise RuntimeError('Credential-bearing content requires review: '+str(relative))
    files = {p.relative_to(ROOT).as_posix(): {'sha256': sha256(p), 'bytes': p.stat().st_size} for p in sorted(paths)}
    total = sum(r['bytes'] for r in files.values()); assert total < 256*1024*1024
    return sorted(paths), files, excluded


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--inventory', action='store_true'); args = parser.parse_args()
    if args.inventory:
        paths, files, excluded = collect_payload(require_audit=False)
        write_json(ROOT/'artifacts/next2/handoff/PAYLOAD_INVENTORY.json', {'status': 'REVIEWABLE_PREFLIGHT',
            'created_utc': utc_now(), 'files': files, 'excluded_reports': excluded,
            'files_count': len(files), 'uncompressed_bytes': sum(v['bytes'] for v in files.values())})
        print('NEXT2 PAYLOAD INVENTORY', len(files), sum(v['bytes'] for v in files.values()), flush=True); return
    timing = require_handoff_time(); guard = SealGuard(ROOT); guard.check_state()
    audit = json.loads((ROOT/'FINAL_AUDIT_NEXT2.json').read_text()); assert audit['status'] == 'PASS' and all(audit['checks'].values())
    for path, digest in audit['evidence_sha256'].items(): assert sha256(ROOT/path) == digest, path
    paths, files, excluded = collect_payload()
    manifest = ROOT/'SOURCE_MANIFEST_NEXT2.json'
    write_json(manifest, {'status': 'FROZEN_PACKAGE_SOURCE_MANIFEST', 'created_utc': utc_now(), 'files': files,
        'files_excluding_this_manifest': len(files), 'uncompressed_bytes': sum(r['bytes'] for r in files.values()),
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'final_audit_sha256': sha256(ROOT/'FINAL_AUDIT_NEXT2.json'), 'clock': timing,
        'excluded_reports': excluded,
        'exclusions': ['raw competition observations and labels', 'real point-level OOF and feature caches', 'credentials and installer logs',
            'environments, Git, prior ZIPs, reproduction workspaces', 'detailed synthetic SHAP rows; grouped summaries retained',
            'unselected learner binaries except the three frozen original references and C205 component'],
        'small_array_scope': 'Selected bootstrap score draws and aggregate synthetic bootstrap summaries only',
        'self_hash': 'This manifest and complete archive are hashed in the external ZIP verification receipt; no recursive self hash.'})
    paths.append(manifest)
    desktop = Path(subprocess.check_output(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command',
        "[Environment]::GetFolderPath('Desktop')"], text=True).strip()).resolve(); assert desktop.is_dir()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination = desktop/('STRUCTURAL_BREAK_GPT_HANDOFF_NEXT2_'+stamp+'.zip'); temporary = destination.with_suffix('.zip.partial')
    assert destination.parent == temporary.parent == desktop and not destination.exists() and not temporary.exists()
    with temporary.open('xb') as stream:
        with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
            for path in paths: archive.write(path, path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None
        names = archive.namelist(); assert len(names) == len(set(names)) == len(paths)
        assert set(names) == {p.relative_to(ROOT).as_posix() for p in paths}
        for path in paths: assert hashlib.sha256(archive.read(path.relative_to(ROOT).as_posix())).hexdigest() == sha256(path)
    temporary.rename(destination)
    receipt = {'status': 'PASS', 'created_utc': utc_now(), 'archive': str(destination), 'bytes': destination.stat().st_size,
        'sha256': sha256(destination), 'entries': len(paths), 'all_member_sha256_verified': True, 'crc_verified': True,
        'manifest_sha256': sha256(manifest), 'raw_competition_data_included': False, 'credentials_included': False,
        'seals_opened': {name: False for name in guard.lock['partitions']}, 'reduced_new_usage': 0, 'public_submission_new': 0,
        'clock': require_handoff_time()}
    write_json(ROOT/'artifacts/next2/handoff/DESKTOP_ZIP_VERIFICATION_NEXT2.json', receipt)
    print('NEXT2 DESKTOP ZIP VERIFIED', str(destination), receipt['sha256'], len(paths), flush=True)


if __name__ == '__main__': main()
