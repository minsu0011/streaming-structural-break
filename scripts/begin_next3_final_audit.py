"""Record the real phase transition only after 11h10 of elapsed research wall."""
from pathlib import Path
import datetime
import os
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, atomic_json, digest, progress, read_json, utc_now
from src.next3.final_core_contract import literal_policy


def main():
    session_path = BASE/'SESSION.json'; session = read_json(session_path)
    if session.get('audit_started_utc'): raise RuntimeError('Preserve the original final audit start')
    if session.get('status') == 'COMPLETE': raise RuntimeError('Completed session cannot restart audit')
    now = datetime.datetime.now(datetime.timezone.utc)
    elapsed = (now-datetime.datetime.fromisoformat(session['started_utc'])).total_seconds()
    if elapsed < 40200: raise RuntimeError('Final audit cannot start before 11h10 research wall')
    literal_policy(read_json(ROOT/'configs/next3/RESEARCH_POLICY_LOCK.json'))
    science = read_json(ROOT/'configs/next3/SCIENTIFIC_ROLE_FREEZE.json')
    assert science['scientific_role_count'] == 2 and not science['new_scientific_search_authorized']
    owned = os.environ.get('NEXT3_JOB_ID')
    for path in (BASE/'jobs').glob('*/STATE.json'):
        state = read_json(path)
        if state['status'] == 'RUNNING' and state['id'] != owned:
            raise RuntimeError('Finish an existing worker before final audit phase transition')
    out = BASE/'final_audit'; out.mkdir(parents=True, exist_ok=True)
    atomic_json(out/'RESEARCH_CLOSURE_SNAPSHOT.json', {'created_utc': now.isoformat(),
        'research_wall_seconds': elapsed, 'session_before': session,
        'accounting_before': read_json(BASE/'runtime/ACCOUNTING.json'),
        'scientific_role_lock_sha256': digest(ROOT/'configs/next3/SCIENTIFIC_ROLE_FREEZE.json'),
        'new_scientific_research_authorized': False,
        'productive_time_not_backfilled': True}, immutable=True)
    session.update(status='FINAL_AUDIT', audit_started_utc=now.isoformat(), research_closed_utc=now.isoformat(),
        research_wall_seconds_at_closure=elapsed, new_scientific_research_authorized=False)
    atomic_json(session_path, session)
    atomic_json(out/'AUDIT_START.json', {'status': 'SEPARATE_FINAL_AUDIT_STARTED',
        'created_utc': now.isoformat(), 'research_wall_seconds': elapsed, 'minimum_audit_wall_seconds': 3000,
        'session_sha256_after_transition': digest(session_path), 'source_sha256': digest(Path(__file__)),
        'scope': 'Actual timestamp, not scheduled/backdated productive time. Final raw training, full tests, actual runtime and delivery remain required.'}, immutable=True)
    progress('NEXT3 separate final audit started', 'Research wall minimum met; fresh raw training and full final checks begin', phase='FINAL_AUDIT')
    print('NEXT3 FINAL AUDIT START', now.isoformat(), 'research wall', elapsed, flush=True)


if __name__ == '__main__': main()
