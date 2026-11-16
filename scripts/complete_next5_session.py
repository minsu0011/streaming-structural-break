"""Close only the owned NEXT5 session after verified Desktop delivery."""
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import psutil
from src.next5.artifacts import BASE, atomic_json, read_json, digest, utc_now, progress
from src.next5.finalization import require_handoff_time
from src.next5.guard import ResearchGuard


def main():
    timing = require_handoff_time()
    session_path = BASE / 'SESSION.json'
    session = read_json(session_path)
    assert session['status'] == 'ACTIVE_FINAL_AUDIT'
    account = read_json(BASE / 'runtime/ACCOUNTING.json')
    assert account['current_job'] is None, 'No active worker may be abandoned'
    pointer_path = BASE / 'handoff/DESKTOP_DELIVERY_POINTER.json'
    pointer = read_json(pointer_path)
    receipt_path = (ROOT / pointer['receipt']).resolve()
    assert receipt_path.is_relative_to((BASE / 'handoff').resolve())
    assert digest(receipt_path) == pointer['sha256']
    receipt = read_json(receipt_path)
    assert receipt['status'] == 'PASS' and receipt['twenty_file_count'] == 20
    for name in ('full_project_archive', 'twenty_file_archive'):
        archive = receipt[name]
        path = Path(archive['path'])
        assert archive['CRC_verified'] and archive['all_member_SHA256_verified']
        assert path.stat().st_size == archive['bytes'] and digest(path) == archive['sha256']
    folder = Path(receipt['twenty_file_folder'])
    assert {p.name for p in folder.iterdir()} == set(receipt['twenty_files'])
    assert len(receipt['twenty_files']) == 20
    for name, expected in receipt['twenty_files'].items():
        assert digest(folder / name) == expected['sha256'] == digest(ROOT / name)
    assert digest(ROOT / 'SOURCE_MANIFEST_NEXT5.json') == receipt['manifest_sha256']
    audit = read_json(ROOT / 'FINAL_AUDIT_NEXT5.json')
    assert audit['status'] == 'PASS' and all(v is True for v in audit['checks'].values())
    assert not read_json(ROOT / 'DEV_CANDIDATES_NEXT5.json')['new_NEXT5_roles']
    guard = ResearchGuard()
    guard.check_state()
    shadow = read_json(BASE / 'shadow/STATE.json')
    assert shadow['status'] == 'CLOSED' and shadow['evaluations_completed'] == 0
    owner = read_json(BASE / 'runtime/SUPERVISOR.json')
    process = psutil.Process(owner['pid'])
    assert process.create_time() == owner['process_create_time']
    assert digest(ROOT / 'src/next5/scheduler.py') == owner['source_sha256']
    out = BASE / 'completion'
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / 'SESSION_BEFORE.json', session, immutable=True)
    stamp = utc_now()
    progress('NEXT5 verified delivery complete', 'All research and separate audit gates passed; owned supervisor will exit normally', phase='DELIVERY')
    atomic_json(session_path, {**session, 'status': 'COMPLETE', 'completed_utc': stamp,
        'automatic_resume_authorized': False, 'final_delivery_receipt': pointer['receipt'],
        'final_delivery_receipt_sha256': pointer['sha256'], 'observed_timing_at_completion': timing})
    # This is an actual process-exit wait, never research-time credit or padding.
    deadline = time.monotonic() + 45
    while process.is_running() and time.monotonic() < deadline:
        assert process.create_time() == owner['process_create_time']
        time.sleep(.25)
    assert not process.is_running(), 'Owned supervisor did not exit normally; do not stop unrelated processes'
    atomic_json(out / 'RESULTS.json', {'created_utc': utc_now(), 'status': 'PASS',
        'completed_utc': stamp, 'session_status': 'COMPLETE', 'owned_supervisor': owner,
        'supervisor_exited_normally': True, 'process_termination_sent': False,
        'Desktop_delivery_receipt_sha256': pointer['sha256'], 'timing_at_completion': timing,
        'archive_and_twenty_files_rechecked': True, 'Shadow_opened': False,
        'new_reduced_uses': 0, 'new_public_submissions': 0, 'new_cloud_submissions': 0,
        'source_sha256': digest(Path(__file__)),
        'scope': 'External post-archive completion receipt; archive content remains unchanged.'}, immutable=True)
    print('NEXT5 COMPLETE; verified Desktop delivery and normal owned-supervisor exit', flush=True)


if __name__ == '__main__':
    main()
