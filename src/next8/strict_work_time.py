"""Completed application work spans, without controller/worker note leases.

This deliberately excludes unrecorded controller implementation and unfinished
work. It is a conservative completion-based clock, not an attention meter.
"""
import math
import time
from collections import Counter
from src.next8.common import ROOT, BASE, read_json, digest, atomic, utc_now, union_seconds
from src.next8.final_reports import merge_intervals, time_contract


def aggregate_completed_work(records, session_start, now, admission=None):
    assert math.isfinite(session_start) and math.isfinite(now) and now >= session_start
    admitted = None if admission is None else admission['admitted_epoch']
    if admitted is not None:
        assert math.isfinite(admitted) and session_start <= admitted <= now
    research = []; audit = []; counted = []; excluded = Counter(); seen = set()
    for record in records:
        key = record['id']
        assert key not in seen, 'Duplicate activity receipt'
        seen.add(key)
        if record['status'] != 'COMPLETE':
            excluded['NOT_SUCCESSFULLY_COMPLETE'] += 1
            continue
        cpu = record['cpu_seconds']
        start, stop = record['started_epoch'], record['ended_epoch']
        assert all(math.isfinite(v) for v in (cpu, start, stop))
        assert cpu >= 0 and session_start <= start <= stop <= now
        if cpu <= .01 or stop == start:
            excluded['NO_SUFFICIENT_RECORDED_PROCESS_CPU'] += 1
            continue
        kind = record['kind']
        if kind in ('SLEEP', 'WAIT', 'IDLE', 'MONITOR', 'CONTROLLER_LEASE'):
            excluded['NON_WORK_KIND'] += 1
            continue
        if kind == 'FINAL_AUDIT':
            assert admitted is not None and start >= admitted, 'Premature final audit activity'
            audit.append([start, stop])
            phase = 'FINAL_AUDIT'
        elif admitted is None or stop <= admitted:
            research.append([start, stop])
            phase = 'PRODUCTIVE_RESEARCH'
        elif start >= admitted:
            # Final tests can invoke helpers which retain their research kind.
            # Their enclosing FINAL_AUDIT operation supplies the audit span.
            excluded['POST_ADMISSION_NON_AUDIT_HELPER'] += 1
            continue
        else:
            raise AssertionError('Research activity crosses phase admission')
        counted.append({'id': key, 'phase': phase, 'kind': kind,
                        'started_epoch': start, 'ended_epoch': stop, 'process_cpu_seconds': cpu})
    research, audit = merge_intervals(research), merge_intervals(audit)
    return {'intervals': research, 'audit_intervals': audit, 'last_epoch': now,
            'productive_seconds': union_seconds(research), 'audit_seconds': union_seconds(audit),
            'counted_activities': counted, 'excluded_activity_counts': dict(excluded)}


def completed_work_snapshot(admission=None):
    session = read_json(BASE / 'SESSION_NEXT8.json')
    records = []; files = {}
    for path in sorted((BASE / 'ledger').glob('*.json')):
        record = read_json(path)
        assert path.stem == record['id']
        # The immutable ledger and terminal activity record must agree.
        sha = digest(path)
        assert digest(BASE / 'runtime/jobs' / path.name) == sha, path.name
        records.append(record)
        files[path.relative_to(BASE).as_posix()] = sha
    result = aggregate_completed_work(records, session['started_epoch'], time.time(), admission)
    return {'status': 'OBSERVED_COMPLETED_WORK', 'timestamp': utc_now(), **result,
            'session_sha256': digest(BASE / 'SESSION_NEXT8.json'), 'ledger_sha256': files,
            'controller_and_worker_completion_note_leases_credited': False,
            'unrecorded_controller_implementation_time_credited': False,
            'unfinished_activity_time_credited': False,
            'parallel_workers_or_nested_activities_summed': False,
            'work_span_semantics': 'Successful recorded computation, fit, test or verification operation; elapsed span includes its required I/O and owned child computation. No note lease, dispatcher polling or gaps between operations.',
            'proof_limit': 'Application operation receipts and observed process CPU, not hardware-level human attention or proof against malicious receipt fabrication.'}


def verify_closure():
    closure = read_json(BASE / 'runtime/FINAL_SOURCE_CLOSURE_NEXT8.json')
    assert closure['status'] == 'FROZEN'
    for name, sha in closure['source_sha256'].items():
        assert digest(ROOT / name) == sha, name


def verify_tasks(*, research_only):
    for task in read_json(BASE / 'runtime/TASK_QUEUE.json'):
        if task['id'] == 'FINAL_PUBLICATION':
            continue
        if research_only and task.get('phase') == 'FINAL_AUDIT':
            continue
        path = BASE / 'runtime/queue' / task['id'] / 'TERMINAL.json'
        assert path.exists() and read_json(path)['status'] == 'COMPLETE', task['id']


def admit_research_completion():
    assert not (BASE / 'runtime/FINAL_AUDIT_ADMISSION.json').exists()
    verify_closure(); verify_tasks(research_only=True)
    result = completed_work_snapshot()
    assert result['productive_seconds'] >= 36000. and result['audit_seconds'] == 0.
    # The original scheduler also requires its own clock threshold. It remains
    # an operational diagnostic and cannot override the stricter lower bound.
    assert read_json(BASE / 'runtime/HEARTBEAT.json')['productive_seconds'] >= 36000.
    certificate = BASE / 'WORK_TIME_RESEARCH_ADMISSION_NEXT8.json'
    atomic(certificate, {**result, 'status': 'PASS', 'research_minimum_seconds': 36000.})
    admission = {'status': 'PASS', 'timestamp': utc_now(), 'admitted_epoch': time.time(),
        'session_sha256': result['session_sha256'], 'productive_seconds_observed': result['productive_seconds'],
        'completed_work_certificate': certificate.name, 'completed_work_certificate_sha256': digest(certificate),
        'clock_basis': 'Completed activity intervals only; no controller/worker completion-note leases'}
    atomic(BASE / 'runtime/FINAL_AUDIT_ADMISSION.json', admission)
    return admission


def admit_final_handoff():
    verify_closure(); verify_tasks(research_only=False)
    admission = read_json(BASE / 'runtime/FINAL_AUDIT_ADMISSION.json')
    assert digest(BASE / admission['completed_work_certificate']) == admission['completed_work_certificate_sha256']
    result = completed_work_snapshot(admission)
    assert result['productive_seconds'] >= 36000. and result['audit_seconds'] >= 3600.
    assert read_json(BASE / 'CORE_ARCHIVE_REPRODUCTION_NEXT8.json')['status'] == 'PASS'
    visual = read_json(BASE / 'FIGURES/VIEW_REVIEW_NEXT8.json')
    assert visual['status'] == 'PASS' and visual['figure_count'] == 8
    certificate = BASE / 'WORK_TIME_HANDOFF_ADMISSION_NEXT8.json'
    atomic(certificate, {**result, 'status': 'PASS', 'separate_audit_minimum_seconds': 3600.})
    receipt = {'status': 'PASS', 'timestamp': utc_now(),
        'productive_seconds_observed': result['productive_seconds'], 'separate_audit_seconds_observed': result['audit_seconds'],
        'completed_work_certificate': certificate.name, 'completed_work_certificate_sha256': digest(certificate),
        'clock_basis': 'Completed activity intervals only; no controller/worker completion-note leases'}
    atomic(BASE / 'runtime/FINAL_HANDOFF_ADMISSION_NEXT8.json', receipt)
    return receipt


def final_work_time_contract(admission):
    path = BASE / admission['completed_work_certificate']
    assert digest(path) == admission['completed_work_certificate_sha256']
    initial = read_json(path)
    assert initial['status'] == 'PASS' and initial['productive_seconds'] >= 36000.
    assert initial['session_sha256'] == digest(BASE / 'SESSION_NEXT8.json')
    snapshot = completed_work_snapshot(admission)
    for name, sha in initial['ledger_sha256'].items():
        assert snapshot['ledger_sha256'][name] == sha, name
    clock = time_contract(snapshot, admission)
    certificate = BASE / 'WORK_TIME_FINAL_CERTIFICATE_NEXT8.json'
    atomic(certificate, {**snapshot, 'status': 'PASS', 'final_time_contract_passed': True})
    return {**clock, 'timestamp': utc_now(),
        'clock_source': 'Immutable completed application activity receipts with process CPU evidence; union of operation spans',
        'controller_work_lease_max_seconds': 0, 'observation_interval_credit_cap_seconds': None,
        'controller_or_worker_completion_note_leases_credited': False,
        'unrecorded_controller_implementation_time_excluded_conservatively': True,
        'unfinished_work_excluded': True, 'completed_work_certificate': certificate.name,
        'completed_work_certificate_sha256': digest(certificate),
        'admission_sha256': digest(BASE / 'runtime/FINAL_AUDIT_ADMISSION.json'),
        'accounting_limits': snapshot['proof_limit']}
