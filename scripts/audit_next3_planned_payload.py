"""Exercise source-copy filters before final audit, without making a ZIP or claim."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, atomic_json, digest, progress, read_json, utc_now
from scripts.package_next3_handoff import snapshot_payload


if __name__ == '__main__':
    directory = BASE/'handoff/maintenance_source_snapshot_01'
    directory.mkdir(parents=True, exist_ok=False)
    atomic_json(directory/'START.json', {'status': 'MAINTENANCE_DRY_RUN', 'created_utc': utc_now(),
        'purpose': 'Exercise the actual source-copy/content filters before the final-audit phase',
        'archive_created': False, 'final_audit_completed': False,
        'script_sha256': digest(Path(__file__))}, immutable=True)
    progress('NEXT3 planned payload filter audit', 'Stable source copies only; no final ZIP, raw reproduction or early final audit')
    files, excluded, missing = snapshot_payload(directory/'payload', require_audit=False)
    for name, record in files.items(): assert digest(directory/'payload'/name) == record['sha256']
    atomic_json(directory/'SOURCE_COPY_MANIFEST.json', {'files': files, 'status': 'MAINTENANCE_DRY_RUN_ONLY'}, immutable=True)
    atomic_json(directory/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(),
        'files_copied_and_rehashed': len(files), 'uncompressed_bytes': sum(v['bytes'] for v in files.values()),
        'protected_old_source_files': len(read_json(BASE/'preflight/FROZEN_REFERENCES.json')['protected_sha256']),
        'excluded_optional_records': excluded, 'required_final_reports_still_missing': missing,
        'source_copy_manifest_sha256': digest(directory/'SOURCE_COPY_MANIFEST.json'),
        'archive_created': False, 'raw_training_reproduction_performed': False, 'final_audit_completed': False,
        'scope': 'Pre-audit maintenance of source packaging only; final fresh snapshot and actual Desktop ZIP verification remain required.'}, immutable=True)
    # Compact receipt is outside the excluded snapshot folder for the final handoff.
    atomic_json(BASE/'maintenance/PLANNED_PAYLOAD_FILTER_AUDIT.json', {
        'status': 'PASS', 'created_utc': utc_now(), 'result_path': (directory/'RESULTS.json').relative_to(ROOT).as_posix(),
        'result_sha256': digest(directory/'RESULTS.json'), 'files_copied_and_rehashed': len(files),
        'uncompressed_bytes': sum(v['bytes'] for v in files.values()), 'archive_created': False,
        'scope': 'Maintenance source-copy dry run, not final audit or actual archive replay'}, immutable=True)
    print('NEXT3 PLANNED SOURCE COPY PASS', len(files), flush=True)
