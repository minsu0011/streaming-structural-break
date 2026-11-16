"""Extract and replay the actual Desktop NEXT3 ZIP, without original project reads."""
from pathlib import Path
import datetime
import shutil
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, read_json, atomic_json, digest, utc_now, progress
from src.next3.archive_contract import extract_verified_archive
from src.next3.path_budget import check_windows_copy_path_budget
from scripts.prepare_next3_independent_reproduction import time_gate


def main():
    timing = time_gate(); receipt_path = BASE/'handoff/DESKTOP_ZIP_VERIFICATION_NEXT3.json'; receipt = read_json(receipt_path)
    assert receipt['status'] == 'PASS' and receipt['actual_archive_replay'] == 'PENDING'
    archive = Path(receipt['archive']); assert digest(archive) == receipt['archive_sha256']
    out = BASE/'handoff/actual_archive_replay'; out.mkdir(parents=True, exist_ok=False)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination = (ROOT.parent/('SB_NEXT3_ZIP_REPLAY_'+stamp)).resolve()
    assert destination.parent == ROOT.parent.resolve() and not destination.exists()
    manifest_path = ROOT/'SOURCE_MANIFEST_NEXT3.json'
    assert digest(manifest_path) == receipt['manifest_sha256']
    manifest = read_json(manifest_path)
    path_budget = check_windows_copy_path_budget(destination,[*manifest['files'],
        'SOURCE_MANIFEST_NEXT3.json','PORTABLE_COPY_MANIFEST_NEXT3.json',
        'ARCHIVED_RUNTIME_RESULT_NEXT3.json','DOCUMENTED_REVIEWER_RESULT_NEXT3.json'])
    atomic_json(out/'START.json', {'created_utc': utc_now(), 'archive_sha256': receipt['archive_sha256'],
        'destination': str(destination), 'windows_path_budget':path_budget,
        'script_sha256': digest(Path(__file__)), 'clock': timing}, immutable=True)
    proof = extract_verified_archive(archive, destination, expected_manifest_sha256=receipt['manifest_sha256'])
    assert proof['archive_sha256'] == receipt['archive_sha256']
    atomic_json(destination/'PORTABLE_COPY_MANIFEST_NEXT3.json', {'files': proof['files'], 'archive_sha256': proof['archive_sha256'],
        'source_root': str(ROOT), 'raw_competition_inputs_included': False}, immutable=True)
    preparation = read_json(BASE/'independent_reproduction/PREPARATION.json')
    assert preparation['status'] == 'PREPARED'
    command = [preparation['python'], '-I', str(destination/'scripts/check_next3_archived_runtime.py'), '--origin', str(ROOT)]
    progress('Actual NEXT3 archive replay', 'Extracted member hashes exact; archived public inference in pinned sibling environment', phase='FINAL_AUDIT')
    with (out/'RUN.log').open('x', encoding='utf-8') as log:
        completed = subprocess.run(command, cwd=destination, stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if completed.returncode:
        atomic_json(out/'FAILURE.json', {'status': 'FAIL', 'created_utc': utc_now(), 'returncode': completed.returncode,
            'log_sha256': digest(out/'RUN.log'), 'all_attempt_files_preserved': True}, immutable=True)
        raise RuntimeError('Actual archive replay failed; preserve all evidence')
    source = destination/'ARCHIVED_RUNTIME_RESULT_NEXT3.json'; report = read_json(source)
    assert report['status'] == 'PASS' and report['archive_sha256'] == receipt['archive_sha256']
    assert report['public_API_prediction_points'] == 98372 and set(report['roles']) == {'DISTILLED', 'PERFORMANCE'}
    assert not any(report[key] for key in ('original_project_access_attempts', 'persistent_write_attempts', 'network_attempts'))
    assert digest(archive) == receipt['archive_sha256']
    shutil.copyfile(source, out/source.name); assert digest(source) == digest(out/source.name)
    result = {'status': 'PASS', 'created_utc': utc_now(), 'archive': str(archive), 'archive_sha256': digest(archive),
        'archive_entries_rehashed': proof['entries'], 'extraction_directory': str(destination),
        'windows_path_budget':path_budget,
        'public_API_prediction_points': report['public_API_prediction_points'],
        'archived_runtime_receipt_sha256': digest(out/source.name), 'original_project_access_attempts': [],
        'raw_competition_observations_used': False, 'new_fits': 0, 'seal_rows': 0, 'clock': time_gate()}
    atomic_json(out/'RESULTS.json', result, immutable=True)
    receipt.update(actual_archive_replay='PASS', actual_archive_replay_sha256=digest(out/'RESULTS.json'),
        final_verification_utc=utc_now(), final_clock=time_gate())
    atomic_json(receipt_path, receipt)
    print('NEXT3 DESKTOP ZIP FULLY VERIFIED', archive, receipt['archive_sha256'], flush=True)


if __name__ == '__main__': main()
