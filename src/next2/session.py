"""Checkpointed monotonic research clock with fail-closed reboot recovery."""
from pathlib import Path
import datetime
import json
import time
import uuid
from src.utils.artifacts import write_json, sha256, utc_now, commit_id

ROOT = Path(__file__).resolve().parents[2]


def clock(root=ROOT):
    session = json.loads((root/'artifacts/next2/SESSION.json').read_text(encoding='utf-8'))
    now = time.monotonic()
    start = session['monotonic_start']
    origin = datetime.datetime.fromisoformat(session['started_utc']).timestamp() - start
    current_origin = time.time() - now
    if abs(current_origin-origin) > 120:
        raise RuntimeError('Clock discontinuity: recover an explicit new active segment from the last checkpoint; never count the reboot/offline gap.')
    active = now-start + sum(s['active_seconds'] for s in session['segments'])
    elapsed = session.get('research_elapsed_at_audit_start',active)
    audit = now-session['audit_monotonic_start'] if 'audit_monotonic_start' in session else 0.
    return {'created_utc': utc_now(), 'monotonic_now': now,
            'elapsed_research_seconds': elapsed, 'minimum_research_seconds': 21600,
            'minimum_research_met': elapsed >= 21600,
            'remaining_research_seconds': max(0.,21600-elapsed),
            'audit_started': 'audit_monotonic_start' in session,
            'elapsed_audit_seconds':audit,'handoff_time_gate_pass':elapsed>=21600 and audit>=2700}


def checkpoint(task, next_tasks, *, artifacts=(), failed_tasks=(), root=ROOT):
    results = list((root/'artifacts/next2/experiments').glob('*/RESULTS.json'))
    selection_path = root/'DEV_CANDIDATES_NEXT2.json'
    if not selection_path.exists():
        selection_path = root/'artifacts/next2/RESEARCH_LEADERS.json'
    selection = json.loads(selection_path.read_text(encoding='utf-8')) if selection_path.exists() else {}
    previous = sorted((root/'checkpoints').glob('NEXT2_*.json'))
    number = max([int(p.stem.split('_')[-1]) for p in previous] + [0])+1
    report = {**clock(root), 'task': task, 'experiment_count': len(results),
              'current_primary': 'J102_SERIAL_VARIANCE_512',
              'current_distilled': selection.get('distilled'),
              'current_performance_best': selection.get('performance'),
              'current_selection_status': selection.get('status','RESEARCH_OPEN'),
              'next_task': list(next_tasks), 'failed_tasks': list(failed_tasks),
              'git_commit': commit_id(),
              'artifact_hashes': {str(p): sha256(root/p) for p in artifacts}}
    # Independent subprocesses may finish simultaneously. Exclusive creation
    # reserves a unique sequence number; no existing checkpoint is replaced.
    while True:
        path = root/f'checkpoints/NEXT2_{number:04d}.json'
        try:
            with path.open('x',encoding='utf-8') as stream:
                json.dump(report,stream,indent=2,ensure_ascii=False,allow_nan=False)
            break
        except FileExistsError:
            number += 1
    # This pointer is advisory; recovery reads the highest valid numbered file.
    latest = root/'artifacts/next2/LATEST_CHECKPOINT.json'
    temporary = latest.with_name(latest.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    try:
        temporary.replace(latest)
    except PermissionError:
        # Windows can briefly deny replacement while another writer replaces
        # the same advisory pointer. The numbered checkpoint is already safe.
        try:
            temporary.unlink(missing_ok=True)
        except PermissionError:
            pass
    return report


def require_research_complete(root=ROOT):
    result = clock(root)
    if not result['minimum_research_met']:
        raise RuntimeError('Six substantive research hours must finish before final audit/handoff.')
    return result


def begin_final_audit(root=ROOT):
    result = require_research_complete(root)
    path = root/'artifacts/next2/SESSION.json'
    session = json.loads(path.read_text(encoding='utf-8'))
    if 'audit_monotonic_start' in session:
        raise RuntimeError('Final audit already started; its clock cannot be reset')
    session.update(status='FINAL_AUDIT',audit_started_utc=utc_now(),
                   audit_monotonic_start=time.monotonic(),
                   research_elapsed_at_audit_start=result['elapsed_research_seconds'])
    write_json(root/'artifacts/next2/RESEARCH_DURATION.json',result)
    write_json(path,session)
    return clock(root)


def require_handoff_time(root=ROOT):
    result = clock(root)
    if not result['handoff_time_gate_pass']:
        raise RuntimeError('Handoff requires six research hours plus at least 45 separate audit minutes')
    return result


def recover_research_from_checkpoint(root=ROOT):
    """Conservatively exclude the entire unobserved gap; never invent credit."""
    path = root/'artifacts/next2/SESSION.json'
    session = json.loads(path.read_text(encoding='utf-8'))
    if 'audit_monotonic_start' in session:
        raise RuntimeError('An interrupted audit needs a separately documented audit recovery')
    checkpoints = sorted((root/'checkpoints').glob('NEXT2_*.json'),reverse=True)
    latest = None
    for p in checkpoints:
        try:
            candidate = json.loads(p.read_text(encoding='utf-8'))
            if candidate['elapsed_research_seconds']>=0:
                latest=(p,candidate)
                break
        except (ValueError,KeyError):
            continue
    if latest is None:
        raise RuntimeError('No valid checkpoint supports any previous active-time credit')
    p,credit = latest
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    write_json(root/f'artifacts/next2/clock_recovery/{stamp}/PREVIOUS_SESSION.json',session)
    original = session.get('original_research_started_utc',session['started_utc'])
    session.update(original_research_started_utc=original,started_utc=utc_now(),monotonic_start=time.monotonic(),
        segments=[{'active_seconds':credit['elapsed_research_seconds'],
                   'checkpoint_path':p.relative_to(root).as_posix(),'checkpoint_sha256':sha256(p),
                   'excluded_gap_start_utc':credit['created_utc'],'excluded_gap_end_utc':utc_now()}])
    write_json(path,session)
    return clock(root)
