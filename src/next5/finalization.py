"""Observed NEXT5 timing gates; no backdating or assumed worker completion."""
from datetime import datetime
import math
import time
from src.next5.artifacts import ROOT, BASE, read_json, atomic_json, digest, utc_now

RESEARCH_MIN_SECONDS = 36000
AUDIT_MIN_SECONDS = 3000


def require_timing(account, session, *, now_wall):
    for key in ('productive_research_seconds', 'audit_productive_seconds', 'unobserved_seconds',
                'wall_seconds', 'wall_timestamp'):
        if type(account[key]) not in (int, float) or not math.isfinite(account[key]) or account[key] < 0:
            raise ValueError('Invalid observed time: ' + key)
    if not math.isfinite(now_wall) or not 0 <= now_wall - account['wall_timestamp'] <= 60:
        raise RuntimeError('Fresh actual supervisor accounting required')
    if account['productive_research_seconds'] < RESEARCH_MIN_SECONDS:
        raise RuntimeError('At least 36000 observed productive research seconds required')
    if not session.get('audit_started_utc'):
        raise RuntimeError('Separate actual final-audit start required')
    audit_wall = account['wall_timestamp'] - datetime.fromisoformat(session['audit_started_utc']).timestamp()
    if audit_wall < AUDIT_MIN_SECONDS or account['audit_productive_seconds'] < AUDIT_MIN_SECONDS:
        raise RuntimeError('At least 3000 separate observed final-audit seconds required')
    if account['audit_productive_seconds'] > audit_wall + 1:
        raise RuntimeError('Audit time predates its actual phase')
    total = sum(account[k] for k in ('productive_research_seconds', 'audit_productive_seconds', 'unobserved_seconds'))
    if total > account['wall_seconds'] + 2:
        raise RuntimeError('Observed accounting exceeds actual elapsed time')
    return {**{k: account[k] for k in ('productive_research_seconds', 'audit_productive_seconds',
        'wall_seconds', 'unobserved_seconds')}, 'accounted_utc': account['timestamp'],
        'audit_started_utc': session['audit_started_utc'], 'audit_wall_seconds': audit_wall,
        'research_minimum_pass': True, 'separate_audit_minimum_pass': True,
        'audit_within_60_minutes_at_check': audit_wall <= 3600, 'no_retroactive_time_credit': True}


def require_handoff_time():
    return require_timing(read_json(BASE / 'runtime/ACCOUNTING.json'), read_json(BASE / 'SESSION.json'), now_wall=time.time())


def begin_audit():
    session_path = BASE / 'SESSION.json'
    session = read_json(session_path)
    if session.get('audit_started_utc') or session['status'] != 'ACTIVE_RESEARCH':
        raise RuntimeError('Only the original active research phase may enter audit once')
    account = read_json(BASE / 'runtime/ACCOUNTING.json')
    if not 0 <= time.time() - account['wall_timestamp'] <= 60:
        raise RuntimeError('Fresh supervisor accounting required')
    if account['productive_research_seconds'] < RESEARCH_MIN_SECONDS:
        raise RuntimeError('The ten-hour observed research minimum is incomplete')
    selection_path = ROOT / 'DEV_CANDIDATES_NEXT5.json'
    selection = read_json(selection_path)
    assert selection['status'] == 'FROZEN_NEXT5_FINAL_ROLES'
    finalist_path = BASE / 'shadow/FINALIST_MANIFEST_NEXT5.json'
    finalist = read_json(finalist_path)
    assert finalist['status'] == 'FROZEN_NEXT5_FINALISTS'
    assert digest(ROOT / 'FINALIST_MANIFEST_NEXT5.json') == digest(finalist_path)
    unresolved, resolved = [], []
    for path in sorted((BASE / 'queue').glob('*.json')):
        job = read_json(path)
        if job['kind'] in ('final_audit', 'delivery'):
            continue
        state_path = BASE / 'jobs' / job['id'] / 'STATE.json'
        state = read_json(state_path) if state_path.exists() else {'status': 'PENDING'}
        if state['status'] == 'COMPLETE' and state.get('exit_code') == 0:
            continue
        resolution_path = BASE / 'failure_resolutions' / (job['id'] + '.json')
        if state['status'] == 'QUARANTINED' and resolution_path.exists():
            resolution = read_json(resolution_path)
            assert resolution['status'] == 'RESOLVED_WITH_PRESERVED_FAILURE'
            assert resolution['quarantined_state_sha256'] == digest(state_path)
            for name, sha in resolution['evidence_sha256'].items():
                assert digest(ROOT / name) == sha
            resolved.append({'job': job['id'], 'resolution_sha256': digest(resolution_path)})
        else:
            unresolved.append({'job': job['id'], 'status': state['status']})
    if unresolved:
        raise RuntimeError('Research or validation jobs remain: ' + str(unresolved))
    out = BASE / 'phase_transitions/RESEARCH_TO_FINAL_AUDIT'
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / 'SESSION_BEFORE.json', session, immutable=True)
    atomic_json(out / 'ACCOUNTING_BEFORE.json', account, immutable=True)
    stamp = utc_now()
    updated = {**session, 'status': 'ACTIVE_FINAL_AUDIT', 'audit_started_utc': stamp,
        'audit_research_seconds_at_transition': account['productive_research_seconds'],
        'final_selection_sha256': digest(selection_path), 'finalist_manifest_sha256': digest(finalist_path)}
    atomic_json(session_path, updated)
    atomic_json(out / 'TRANSITION.json', {'status': 'PASS', 'created_utc': stamp,
        'research_productive_seconds_observed_before_transition': account['productive_research_seconds'],
        'audit_started_utc': stamp, 'audit_time_backdated': False, 'pending_required_jobs': 0,
        'resolved_preserved_failures': resolved, 'selection_sha256': digest(selection_path),
        'source_sha256': digest(ROOT / 'src/next5/finalization.py')}, immutable=True)
    return updated
