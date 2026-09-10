"""Exclude explicitly audited interruptions from the unchanged wall clock.

The original ResearchClock implementation is part of the frozen late-research
source contract. This additive gate preserves it and prevents interrupted wall
time from satisfying the user's ten substantive research hours.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
from src.next.research_clock import ResearchClock, ResearchTimeError
from src.utils.artifacts import sha256


def adjusted_seconds(start, end, intervals):
    """Validate nonoverlapping aware-UTC intervals and subtract their duration."""
    def parse(value):
        value = datetime.fromisoformat(value) if isinstance(value, str) else value
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise ResearchTimeError('Explicit timezone-aware time required')
        return value.astimezone(timezone.utc)
    start, end = parse(start), parse(end)
    if end < start:
        raise ResearchTimeError('Research end precedes the original start')
    previous, excluded = start, 0.
    for left, right in sorted((parse(left), parse(right)) for left, right in intervals):
        if left < previous or right <= left or right > end:
            raise ResearchTimeError('Overlapping, empty or out-of-window interruption')
        excluded += (right-left).total_seconds()
        previous = right
    wall = (end-start).total_seconds()
    return {'wall_seconds': wall, 'excluded_seconds': excluded,
        'substantive_research_seconds': wall-excluded}


def snapshot(root):
    root = Path(root)
    clock = ResearchClock(root).snapshot()  # Retain reboot/drift detection.
    session_path = root/'artifacts/next/SESSION_NEXT.json'
    session = json.loads(session_path.read_text(encoding='utf-8'))
    folder = root/'artifacts/next/scheduler'
    audit_path = folder/'FINAL_AUDIT_START.json'
    end = json.loads(audit_path.read_text(encoding='utf-8'))['created_utc'] if audit_path.exists() else datetime.now(timezone.utc).isoformat()
    files = {session_path.relative_to(root).as_posix(): sha256(session_path)}
    intervals, records = [], []
    for path in sorted(folder.glob('recovery_*/RESULTS.json')):
        record = json.loads(path.read_text(encoding='utf-8'))
        if record['status'] != 'PASS' or not record['session_start_unchanged'] or record['session_sha256'] != sha256(session_path):
            raise ResearchTimeError('Invalid interruption recovery evidence')
        proof = root/record['last_completed_work_evidence']
        old_anchor = path.parent/'MONOTONIC_ANCHOR_BEFORE.json'
        if sha256(proof) != record['last_completed_work_sha256'] or sha256(old_anchor) != record['old_anchor_sha256']:
            raise ResearchTimeError('Recovery source evidence changed')
        left, right = record['last_conservatively_verified_work_utc'], record['first_resumed_work_utc']
        duration = adjusted_seconds(left, right, [])['wall_seconds']
        if abs(duration-record['excluded_seconds']) > 1e-6:
            raise ResearchTimeError('Recovery exclusion duration differs')
        if records and records[-1]['new_anchor_sha256'] != record['old_anchor_sha256']:
            raise ResearchTimeError('Recovery anchor chain differs')
        intervals.append((left, right))
        records.append(record)
        for item in (path, proof, old_anchor):
            files[item.relative_to(root).as_posix()] = sha256(item)
    anchor = folder/'MONOTONIC_ANCHOR.json'
    if records and sha256(anchor) != records[-1]['new_anchor_sha256']:
        raise ResearchTimeError('Current anchor differs from the explicit recovery')
    files[anchor.relative_to(root).as_posix()] = sha256(anchor)
    if audit_path.exists():
        files[audit_path.relative_to(root).as_posix()] = sha256(audit_path)
    result = adjusted_seconds(session['research_start_utc'], end, intervals)
    result.update(original_start_utc=session['research_start_utc'], research_end_utc=end,
        minimum_research_seconds=session['minimum_research_seconds'],
        substantive_minimum_met=result['substantive_research_seconds'] >= session['minimum_research_seconds'],
        remaining_substantive_research_seconds=max(0., session['minimum_research_seconds']-result['substantive_research_seconds']),
        final_audit_started=clock['final_audit_started'], exclusions=intervals,
        evidence_files_sha256=files)
    return result


def verify_final_duration(root):
    root = Path(root)
    path = root/'artifacts/next/scheduler/SUBSTANTIVE_RESEARCH_DURATION.json'
    report = json.loads(path.read_text(encoding='utf-8'))
    current = snapshot(root)
    if report['status'] != 'PASS' or not current['final_audit_started'] or not current['substantive_minimum_met']:
        raise ResearchTimeError('Ten interruption-adjusted research hours and explicit audit start required')
    if report['active_clock_source_sha256'] != sha256(Path(__file__)):
        raise ResearchTimeError('Substantive-duration implementation changed')
    for key in ('original_start_utc', 'research_end_utc', 'wall_seconds', 'excluded_seconds', 'substantive_research_seconds', 'evidence_files_sha256'):
        if report[key] != current[key]:
            raise ResearchTimeError('Substantive-duration audit changed: '+key)
    return report
