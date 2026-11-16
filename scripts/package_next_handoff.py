"""Build and verify the requested Desktop ZIP only after both duration gates.

Explicit source/report/model allowlists exclude raw inputs, feature/OOF caches,
environments, reproduction workspaces, credentials and earlier archives.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import zipfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.research_clock import ResearchClock
from src.next.guard import SealGuard
from src.utils.artifacts import write_json, sha256, utc_now

REQUIRED = ('00_GPT_READ_FIRST_NEXT.md', '01_GPT_HANDOFF_NEXT.md', 'SEAL_PARTITION_500x4_LOCK.json',
    'SEAL_GUARD_REPORT.json', 'ROBUSTNESS_SPLIT_LOCK.json', 'CURRENT_TOP_FORENSIC.md',
    'CURRENT_TOP_DISAGREEMENT.csv', 'PAIRED_BOOTSTRAP_RESULTS.csv', 'HISTORICAL_REGIME_POLICY.md',
    'HISTORICAL_REGIME_RESULTS.csv', 'MECHANISM_SIGNATURE_POLICY.md', 'MECHANISM_FRACTURE_RESULTS.csv',
    'WHITENING_RESULTS.csv', 'INNOVATION_NORMALIZATION_RESULTS.csv', 'SEQUENTIAL_STAT_RESULTS.csv',
    'GLR_RESULTS.csv', 'BAYES_LITE_RESULTS.csv', 'RANKING_OBJECTIVE_RESULTS.csv', 'WEIGHTING_RESULTS_NEXT.csv',
    'FEATURE_ABLATION_NEXT.csv', 'MODEL_RESULTS_NEXT.csv', 'CV_RESULTS_NEXT.csv', 'TIME_FRACTURE_NEXT.csv',
    'BREAK_AGE_NEXT.csv', 'LATENCY_NEXT.json', 'DETERMINISM_NEXT.json', 'REPRODUCTION_NEXT.json',
    'EXPERIMENT_LEDGER.csv', 'DEV_CANDIDATES_NEXT.json', 'SEAL_OPEN_RECOMMENDATION.md',
    'LIMITATIONS_NEXT.md', 'NEXT_SESSION_PLAN.md')
FORBIDDEN_PARTS = {'.git', '.crunchdao', '.venv', '.repro-venv', '.figure-venv', '__pycache__',
    'reproduction_workspaces', 'portable_workspaces', '.next-portable-v2', '.next-portable-v3', 'independent_next', 'node_modules'}
REPORT_SUFFIXES = {'.json', '.jsonl', '.csv', '.md', '.txt', '.xml', '.png', '.svg', '.py'}


def ordinary_files(directory):
    """Do not descend into junctions, symlinks or excluded directory names."""
    directory = Path(directory)
    for parent, dirs, files in os.walk(directory, followlinks=False):
        parent = Path(parent)
        dirs[:] = [name for name in dirs if name not in FORBIDDEN_PARTS and not (parent/name).is_symlink() and not (parent/name).is_junction()]
        for name in files:
            path = parent/name
            if not path.is_symlink() and not path.is_junction():
                yield path


def collect_payload(selection):
    """One reviewed allowlist shared by preflight and the duration-gated ZIP."""
    if selection['status'] != 'FROZEN_DEV_SELECTION' or not 1 <= len(selection['candidates']) <= 3:
        raise RuntimeError('Exactly one to three frozen final DEV candidates required')
    missing = [name for name in REQUIRED if not (ROOT/name).is_file()]
    if missing:
        raise RuntimeError('Required final reports missing: '+', '.join(missing))
    paths = {ROOT/name for name in REQUIRED}
    paths.update(ROOT/f'SEAL_{letter}_IDS.txt' for letter in 'ABCD')
    paths.update(ROOT/name for name in ('main.py', 'next_main.py', 'next_readonly_main.py', 'next_registered_main.py', 'next_serial_main.py',
        'requirements-lock.txt', 'requirements-next-figures-lock.txt', '.gitattributes', 'REDUCED_TEST_USAGE_LEDGER.json') if (ROOT/name).is_file())
    for folder in ('src', 'scripts', 'tests', 'vendor/official'):
        paths.update(path for path in ordinary_files(ROOT/folder) if path.suffix in {'.py', '.md', '.txt'})
    # Preserved legacy feature modules load their JSON defaults at import time.
    # They are runtime dependencies of the NEXT adapter, including features,
    # arch_input, historical_context/quantiles and magnitude/rank policies.
    paths.update(path for path in ordinary_files(ROOT/'configs') if path.suffix == '.json')
    paths.update(path for path in ordinary_files(ROOT/'docs') if path.suffix == '.md' and ('NEXT' in path.name.upper() or 'STRUCTURAL' in path.name.upper()))
    paths.update((ROOT/'checkpoints').glob('STAGE_NEXT_CHECKPOINT_*.json'))
    # Frozen group membership is metadata required to recreate the DEV boundary.
    paths.update(ROOT/name for name in ('artifacts/validation/SPLIT_LOCK_REAL.json', 'artifacts/validation/GROUP_ASSIGNMENTS.parquet'))
    selected_ids = {row['experiment_id'] for row in selection['candidates']}
    for path in ordinary_files(ROOT/'artifacts/next'):
        relative = path.relative_to(ROOT)
        if 'handoff' in relative.parts:
            continue
        if path.name == 'PIP_INSTALL.txt':
            # Keep installer diagnostics locally. The portable version/check
            # receipts suffice for handoff; installer logs can contain index
            # configuration and are not an important scientific result.
            continue
        if path.name == 'SERIES.csv' and 'alarm_calibration' in relative.parts:
            # These eleven diagnostic tables retain each DEV series' original
            # tau label. Keep aggregates and provenance receipts in the ZIP.
            continue
        if path.suffix in REPORT_SUFFIXES:
            paths.add(path)
        elif path.suffix == '.npy' and 'forensic' in relative.parts and any(eid in relative.parts for eid in selected_ids) and path.name.startswith('fold_') and path.stat().st_size <= 20000:
            # These small arrays contain 2000 bootstrap statistics, not raw
            # observations, features, labels or point-level OOF predictions.
            paths.add(path)
    model_files = {}
    for row in selection['candidates']:
        directory = ROOT/row['delivery_model_directory']
        if not directory.resolve().is_relative_to((ROOT/'artifacts/next/deployment/final').resolve()):
            raise RuntimeError('A final delivery model path escapes the intended model directory')
        for name in ('model.joblib', 'model.json'):
            path = directory/name
            expected = row['model_sha256'] if name == 'model.joblib' else row['delivery_descriptor_sha256']
            if sha256(path) != expected:
                raise RuntimeError('Final delivery artifact changed')
            paths.add(path)
            model_files[path.relative_to(ROOT).as_posix()] = expected
    manifest_path = ROOT/'SOURCE_MANIFEST_NEXT.json'
    paths.discard(manifest_path)
    paths = sorted(paths, key=lambda path: path.relative_to(ROOT).as_posix())
    for path in paths:
        relative = path.relative_to(ROOT)
        if any(part in FORBIDDEN_PARTS for part in relative.parts) or relative.parts[0] == 'data' or path.suffix.lower() in ('.zip', '.npz', '.pkl', '.pickle', '.pyc', '.nbi', '.nbc'):
            raise RuntimeError('A forbidden raw/cache/environment payload reached the allowlist: '+str(relative))
        if any(word in path.name.lower() for word in ('credential', 'access_token', 'refresh_token', 'private_key')):
            raise RuntimeError('A credential-like filename reached the allowlist')
        if path.suffix == '.parquet' and relative.as_posix() != 'artifacts/validation/GROUP_ASSIGNMENTS.parquet':
            raise RuntimeError('Unexpected parquet in final package')
        if path.suffix == '.joblib' and relative.as_posix() not in model_files:
            raise RuntimeError('Unselected model reached the package')
        if path.suffix == '.csv':
            with path.open(encoding='utf-8-sig', newline='') as stream:
                columns = set(next(csv.reader(stream), []))
            if columns.intersection({'id', 'dataset_id'}) and columns.intersection({'tau', 'target', 'break_time', 'break_index'}):
                raise RuntimeError('A per-series raw-label table reached the package: '+str(relative))
        if path.suffix == '.json':
            pending = [json.loads(path.read_text(encoding='utf-8-sig'))]
            while pending:
                value = pending.pop()
                if isinstance(value, dict):
                    if isinstance(value.get('historical'), list) and isinstance(value.get('online'), list):
                        raise RuntimeError('An observation-bearing inference fixture reached the report allowlist: '+str(relative))
                    if set(value).intersection({'id', 'dataset_id'}) and set(value).intersection({'tau', 'target', 'break_time', 'break_index'}):
                        raise RuntimeError('A per-series raw-label record reached the package: '+str(relative))
                    pending.extend(value.values())
                elif isinstance(value, list):
                    pending.extend(value)
    files = {path.relative_to(ROOT).as_posix(): {'sha256': sha256(path), 'bytes': path.stat().st_size} for path in paths}
    total_bytes = sum(row['bytes'] for row in files.values())
    if total_bytes > 512*1024*1024:
        raise RuntimeError('Important-results package unexpectedly exceeds 512 MiB; review the allowlist')
    return paths, files, model_files, total_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-report', default='artifacts/next/final_audit/RESULTS.json')
    args = parser.parse_args()
    clock = ResearchClock(ROOT).assert_handoff_ready()
    guard = SealGuard(ROOT)
    guard.check_state()
    audit_path = ROOT/args.audit_report
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    if audit['status'] != 'PASS' or not audit['checks'] or not all(value is True for value in audit['checks'].values()):
        raise RuntimeError('Completed final audit required before the Desktop ZIP')
    for relative, digest in audit['evidence_files_sha256'].items():
        if sha256(ROOT/relative) != digest:
            raise RuntimeError('Final audited input changed before packaging: '+relative)
    selection = json.loads((ROOT/'DEV_CANDIDATES_NEXT.json').read_text(encoding='utf-8'))
    paths, files, model_files, total_bytes = collect_payload(selection)
    manifest_path = ROOT/'SOURCE_MANIFEST_NEXT.json'
    write_json(manifest_path, {'status': 'FROZEN_PACKAGE_SOURCE_MANIFEST', 'created_utc': utc_now(),
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'clock': clock, 'files': files, 'files_excluding_this_manifest': len(files), 'uncompressed_bytes_excluding_this_manifest': total_bytes,
        'selected_model_files_sha256': model_files, 'final_audit_sha256': sha256(audit_path),
        'exclusions': ['all raw observations and raw targets, including per-series alarm tables retaining tau', 'feature and point-level OOF caches', 'virtual environments',
            'reproduction workspace trees', 'credentials and account configuration', 'Git database', 'unselected binary models', 'previous ZIP files'],
        'bootstrap_array_scope': 'Only selected-candidate small arrays of bootstrap statistics; no point-level predictions or raw observations.',
        'self_hash': 'Excluded to avoid a recursive manifest; the external ZIP receipt hashes this manifest and the full ZIP.'})
    paths.append(manifest_path)
    desktop_text = subprocess.check_output(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', "[Environment]::GetFolderPath('Desktop')"], text=True).strip()
    desktop = Path(desktop_text).resolve()
    if not desktop.is_dir():
        raise RuntimeError('The actual Windows Desktop folder was not found')
    timestamp = utc_now().replace('-', '').replace(':', '').split('.')[0].replace('+0000', '')+'Z'
    destination = desktop/('STRUCTURAL_BREAK_GPT_HANDOFF_NEXT_'+timestamp+'.zip')
    temporary = destination.with_suffix('.zip.partial')
    if temporary.exists() or destination.exists():
        raise RuntimeError('Refusing to overwrite an existing Desktop archive')
    if destination.parent.resolve() != desktop or temporary.parent.resolve() != desktop:
        raise RuntimeError('Archive path escaped the explicitly selected Desktop folder')
    with temporary.open('xb') as stream:
        with zipfile.ZipFile(stream, mode='w', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
            for path in paths:
                archive.write(path, path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('Desktop archive CRC verification failed')
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != {path.relative_to(ROOT).as_posix() for path in paths}:
            raise RuntimeError('Desktop archive path coverage differs from the source allowlist')
        for path in paths:
            actual = hashlib.sha256(archive.read(path.relative_to(ROOT).as_posix())).hexdigest()
            if actual != sha256(path):
                raise RuntimeError('Desktop archive member does not match its source bytes')
    temporary.rename(destination)
    receipt = {'status': 'PASS', 'created_utc': utc_now(), 'archive': str(destination), 'bytes': destination.stat().st_size,
        'sha256': sha256(destination), 'entries': len(paths), 'all_member_sha256_verified': True,
        'crc_verified': True, 'manifest_sha256': sha256(manifest_path), 'clock': ResearchClock(ROOT).assert_handoff_ready(),
        'raw_data_included': False, 'credentials_included': False, 'seal_opened': {letter: False for letter in 'ABCD'},
        'reduced_new_uses': 0, 'public_submissions': 0}
    write_json(ROOT/'artifacts/next/handoff/DESKTOP_ZIP_VERIFICATION_NEXT.json', receipt)
    print('DESKTOP NEXT ZIP VERIFIED', destination, receipt['sha256'], receipt['entries'], flush=True)


if __name__ == '__main__':
    main()
