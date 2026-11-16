"""Run the documented reviewer command from the actual verified archive extraction."""
from pathlib import Path
import datetime
import shutil
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, atomic_json, digest, progress, read_json, utc_now
from src.next3.final_validation import final_clock, successful_receipt


def main():
    timing = final_clock(read_json(BASE/'SESSION.json'), datetime.datetime.now(datetime.timezone.utc), require_complete_duration=True)
    pointer_path = BASE/'handoff/DESKTOP_ZIP_VERIFICATION_NEXT3.json'; pointer = successful_receipt(pointer_path)
    if pointer['actual_archive_replay'] != 'PASS': raise RuntimeError('Actual origin-denied archive replay must pass first')
    actual_path = BASE/'handoff/actual_archive_replay/RESULTS.json'; actual = successful_receipt(actual_path)
    if digest(actual_path) != pointer['actual_archive_replay_sha256']: raise RuntimeError('Actual archive replay pointer differs')
    archive = Path(pointer['archive'])
    if digest(archive) != pointer['archive_sha256']: raise RuntimeError('Actual archive changed before reviewer command')
    extraction = Path(actual['extraction_directory']).resolve()
    if extraction.parent != ROOT.parent.resolve() or not extraction.name.startswith('SB_NEXT3_ZIP_REPLAY_'):
        raise RuntimeError('Verified actual archive extraction required')
    manifest = read_json(extraction/'SOURCE_MANIFEST_NEXT3.json')
    if digest(extraction/'SOURCE_MANIFEST_NEXT3.json') != pointer['manifest_sha256']: raise RuntimeError('Extracted manifest changed')
    script_name = 'scripts/replay_next3_portable.py'
    if digest(extraction/script_name) != manifest['files'][script_name]['sha256']:
        raise RuntimeError('Reviewer command differs from actual archive source')
    out = BASE/'handoff/reviewer_command'; out.mkdir(parents=True, exist_ok=False)
    preparation = successful_receipt(BASE/'independent_reproduction/PREPARATION.json', statuses=('PREPARED',))
    command = [preparation['python'], '-I', str(extraction/script_name), '--output', 'DOCUMENTED_REVIEWER_RESULT_NEXT3.json']
    atomic_json(out/'START.json', {'created_utc': utc_now(), 'clock': timing, 'command': command,
        'archive_sha256': pointer['archive_sha256'], 'actual_archive_replay_sha256': digest(actual_path),
        'script_sha256': digest(Path(__file__))}, immutable=True)
    progress('Documented NEXT3 reviewer command', 'Run raw-free 98,372-point replay from actual extracted archive source', phase='FINAL_AUDIT')
    with (out/'RUN.log').open('x', encoding='utf-8') as stream:
        completed = subprocess.run(command, cwd=extraction, stdout=stream, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if completed.returncode:
        atomic_json(out/'FAILURE.json', {'status': 'FAIL', 'created_utc': utc_now(),
            'returncode': completed.returncode, 'log_sha256': digest(out/'RUN.log'), 'all_attempt_files_preserved': True}, immutable=True)
        raise RuntimeError('Documented reviewer command failed; original attempt preserved')
    path = extraction/'DOCUMENTED_REVIEWER_RESULT_NEXT3.json'; result = successful_receipt(path)
    if result['public_API_prediction_points'] != 98372 or set(result['roles']) != {'DISTILLED', 'PERFORMANCE'}:
        raise RuntimeError('Reviewer command did not replay both complete model roles')
    if result['source_manifest_sha256'] != pointer['manifest_sha256']:
        raise RuntimeError('Reviewer command used a different source manifest')
    if result['new_fits'] or result['new_quality_scores'] or result['raw_competition_inputs_needed']:
        raise RuntimeError('Reviewer replay escaped its raw-free zero-fit scope')
    if any(result[key] for key in ('persistent_write_attempts_during_inference', 'network_attempts', 'competition_data_access_attempts')):
        raise RuntimeError('Reviewer command reported forbidden access')
    if digest(archive) != pointer['archive_sha256']: raise RuntimeError('Archive changed during reviewer command')
    shutil.copyfile(path, out/path.name)
    if digest(path) != digest(out/path.name): raise RuntimeError('Reviewer receipt copy differs')
    atomic_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(), 'archive_sha256': pointer['archive_sha256'],
        'extraction_directory': str(extraction), 'public_API_prediction_points': 98372,
        'documented_command_receipt_sha256': digest(out/path.name), 'command': command,
        'actual_Desktop_archive_used': True, 'new_fits': 0, 'seal_rows': 0, 'clock': timing,
        'script_sha256': digest(Path(__file__)),
        'scope': 'The documented user command was executed from the already independently verified actual archive extraction.'}, immutable=True)
    pointer.update(documented_reviewer_command='PASS', documented_reviewer_command_sha256=digest(out/'RESULTS.json'),
        documented_reviewer_command_verified_utc=utc_now())
    atomic_json(pointer_path, pointer)
    print('NEXT3 DOCUMENTED REVIEWER COMMAND PASS 98372 POINTS', flush=True)


if __name__ == '__main__': main()
