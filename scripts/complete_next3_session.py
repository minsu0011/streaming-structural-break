"""Mark completion only after core audit, actual ZIP, reviewer command and top 20 pass."""
from pathlib import Path
import datetime
import os
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, atomic_json, digest, progress, read_json, utc_now
from src.next3.final_source_contract import assert_source_snapshot
from src.next3.final_validation import all_true, final_clock, successful_receipt
from src.next3.distribution_validation import distribution_checks
from src.next.guard import SealGuard
from scripts.verify_next3_old_desktop_preservation import verify_existing_desktop_deliveries


def main():
    session_path = BASE/'SESSION.json'; session = read_json(session_path)
    if session['status'] == 'COMPLETE': raise RuntimeError('Preserve an earlier session completion')
    timing = final_clock(session, datetime.datetime.now(datetime.timezone.utc), require_complete_duration=True)
    audit_path = ROOT/'FINAL_AUDIT_NEXT3.json'; audit = successful_receipt(audit_path)
    all_true(audit['checks'])
    if audit['final_duration_closure'] is not True: raise RuntimeError('Final core duration closure required')
    source_lock_path = BASE/'final_audit/FINAL_EXECUTION_SOURCE_LOCK.json'
    assert_source_snapshot(ROOT, read_json(source_lock_path))
    pointer_path = BASE/'handoff/DESKTOP_ZIP_VERIFICATION_NEXT3.json'; pointer = successful_receipt(pointer_path)
    if pointer['actual_archive_replay'] != 'PASS' or pointer.get('documented_reviewer_command') != 'PASS':
        raise RuntimeError('Actual archive and documented reviewer replay must both pass')
    archive = Path(pointer['archive'])
    if digest(archive) != pointer['archive_sha256']: raise RuntimeError('Final Desktop archive changed')
    actual_path = BASE/'handoff/actual_archive_replay/RESULTS.json'; actual = successful_receipt(actual_path)
    command_path = BASE/'handoff/reviewer_command/RESULTS.json'; command = successful_receipt(command_path)
    if digest(actual_path) != pointer['actual_archive_replay_sha256'] or digest(command_path) != pointer['documented_reviewer_command_sha256']:
        raise RuntimeError('Final archive replay pointers differ')
    if not actual['public_API_prediction_points'] == command['public_API_prediction_points'] == 98372:
        raise RuntimeError('Both final replay scopes must be complete')
    top20_path = BASE/'handoff/DESKTOP_TOP20_VERIFICATION_NEXT3.json'; top20 = successful_receipt(top20_path)
    final_distribution_checks = distribution_checks(pointer, actual, command, top20)
    folder = Path(top20['directory'])
    if top20['file_count'] != 20 or len(list(folder.iterdir())) != 20:
        raise RuntimeError('Desktop review folder must contain exactly 20 files')
    for name, record in top20['files'].items():
        path = folder/name
        if not path.is_file() or path.is_symlink() or path.is_junction() or digest(path) != record['sha256']:
            raise RuntimeError('Desktop review file changed: '+name)
    if top20['archive_sha256'] != pointer['archive_sha256']:
        raise RuntimeError('Desktop folder references a different archive')
    if digest(folder/'DESKTOP_ZIP_VERIFICATION_NEXT3.json') != digest(pointer_path):
        raise RuntimeError('The twentieth file must be the actual current verified archive receipt')
    guard = SealGuard(ROOT); guard.check_state()
    old_desktop = verify_existing_desktop_deliveries(ROOT)
    if any(item['opened'] for item in guard.lock['partitions'].values()): raise RuntimeError('All four seals must remain unopened')
    owned = os.environ.get('NEXT3_JOB_ID')
    for path in (BASE/'queue').glob('*.json'):
        job = read_json(path)
        if job['id'] == owned: continue
        state_path = BASE/'jobs'/job['id']/'STATE.json'
        if not state_path.exists() or read_json(state_path)['status'] not in ('COMPLETE', 'QUARANTINED'):
            raise RuntimeError('A registered task is still unresolved: '+job['id'])
    paths = [audit_path, pointer_path, actual_path, command_path, top20_path, source_lock_path]
    completed_utc = utc_now()
    result = {'status': 'PASS', 'created_utc': completed_utc, 'SESSION_COMPLETE': True,
        'conclusion': 'NEXT3_ENGINEERING_IMPROVED', 'clock': timing,
        'observed_accounting': read_json(BASE/'runtime/ACCOUNTING.json'),
        'archive': str(archive), 'archive_sha256': pointer['archive_sha256'],
        'desktop_top20_directory': str(folder), 'desktop_top20_files': 20,
        'both_actual_archive_and_documented_reviewer_replays_passed': True,
        'distribution_checks': final_distribution_checks,
        'earlier_Desktop_deliveries_verified_after_new_delivery': old_desktop,
        'scientific_quality_delta': 0.0, 'scientific_promotions': 0,
        'scientific_fits': 129, 'separate_raw_reproduction_fits': 3, 'total_completed_fits': 132,
        'seals_opened': {name: False for name in guard.lock['partitions']},
        'reduced_new_usage': 0, 'public_submission_new': 0, 'cloud_submission_new': 0,
        'evidence_sha256': {path.relative_to(ROOT).as_posix(): digest(path) for path in paths},
        'source_sha256': digest(Path(__file__)),
        'timing_scope': 'Actual closure after final artifact verification. Unobserved intervals remain excluded from productive research/audit.'}
    atomic_json(BASE/'handoff/SESSION_COMPLETION.json', result, immutable=True)
    session.update(status='COMPLETE', completed_utc=completed_utc,
        conclusion='NEXT3_ENGINEERING_IMPROVED', completion_receipt_sha256=digest(BASE/'handoff/SESSION_COMPLETION.json'))
    atomic_json(session_path, session)
    progress('NEXT3 session complete', 'Actual Desktop ZIP, independent archive replay, reviewer command and exactly 20 core files all verified', phase='FINAL_AUDIT')
    print('NEXT3 SESSION COMPLETE', archive, folder, flush=True)


if __name__ == '__main__': main()
