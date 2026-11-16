"""Strict, small final-audit primitives with explicit evidence boundaries."""
from pathlib import Path
import datetime
import math
import xml.etree.ElementTree as ET
from src.next3.artifacts import digest, read_json


def all_true(checks):
    if not checks or any(type(value) is not bool for value in checks.values()):
        raise ValueError('Nonempty literal boolean audit checks required')
    failed = [name for name, value in checks.items() if not value]
    if failed: raise RuntimeError('Failed audit checks: '+', '.join(failed))


def successful_receipt(path, *, statuses=('PASS',)):
    record = read_json(path)
    if record.get('status') not in statuses:
        raise RuntimeError('Unaccepted evidence status: '+str(path))
    return record


def final_clock(session, now, *, require_complete_duration=False):
    started = datetime.datetime.fromisoformat(session['started_utc'])
    audit = datetime.datetime.fromisoformat(session['audit_started_utc'])
    if any(value.tzinfo is None for value in (started, audit, now)):
        raise ValueError('Timezone-aware audit clocks required')
    research = (audit-started).total_seconds(); elapsed = (now-started).total_seconds()
    audit_elapsed = (now-audit).total_seconds()
    if research < 40200 or audit_elapsed < 0:
        raise RuntimeError('Separate NEXT3 final audit must follow the 11h10 research gate')
    if require_complete_duration and audit_elapsed < 3000:
        raise RuntimeError('Final core audit closure requires 50 minutes of separately elapsed audit time')
    return {'started_utc': started.isoformat(), 'audit_started_utc': audit.isoformat(), 'checked_utc': now.isoformat(),
        'wall_seconds': elapsed, 'research_wall_seconds': research, 'audit_wall_seconds': audit_elapsed,
        'minimum_research_wall_met': research >= 40200, 'minimum_audit_wall_met': audit_elapsed >= 3000,
        'productive_time_authority': 'Supervisor ACCOUNTING/HEARTBEAT observed intervals; elapsed wall is not relabeled productive.'}


def tests_proof(path, *, minimum_count, required_classnames, not_before_timestamp):
    path = Path(path)
    if path.stat().st_mtime < not_before_timestamp:
        raise RuntimeError('A fresh final-audit test run is required')
    root = ET.parse(path).getroot(); suites = list(root.iter('testsuite')); cases = list(root.iter('testcase'))
    if not suites or not cases:
        raise ValueError('Test XML lacks suites or test cases')
    count = sum(int(suite.attrib['tests']) for suite in suites)
    if count != len(cases) or count < minimum_count:
        raise RuntimeError('Final test suite is incomplete')
    if any(int(suite.attrib.get(key, 0)) != 0 for suite in suites for key in ('failures', 'errors', 'skipped')):
        raise RuntimeError('Final tests must pass without errors, failures or skips')
    if any(list(case.iter(tag)) for case in cases for tag in ('failure', 'error', 'skipped')):
        raise RuntimeError('Nested testcase failures contradict suite counters')
    observed = {case.attrib.get('classname') for case in cases}
    if not set(required_classnames).issubset(observed):
        raise RuntimeError('Required scorer or deployment test classes are absent')
    seconds = sum(float(suite.attrib.get('time', 0)) for suite in suites)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError('Invalid test duration')
    return {'status': 'PASS', 'tests': count, 'seconds': seconds,
        'required_classnames': sorted(required_classnames), 'report_sha256': digest(path)}


def prepared_table_proof(root, prepared):
    root = Path(root); prepared = Path(prepared)
    record = successful_receipt(prepared/'RESULTS.json', statuses=('PREPARED_CLOSED_SCIENTIFIC_TABLES_NOT_FINAL_DELIVERY',))
    retained = successful_receipt(prepared/'INPUT_RETENTION.json')
    if digest(prepared/'RESULTS.json') != retained['prepared_receipt_sha256']:
        raise RuntimeError('Prepared table receipt changed after input retention')
    for name, value in record['files'].items():
        if digest(prepared/name) != value['sha256']: raise RuntimeError('Prepared output bytes changed: '+name)
    mapping = retained['mutable_source_snapshots']
    if not set(mapping).issubset(record['source_evidence_sha256']):
        raise RuntimeError('Retention mapping introduces a new source')
    for name, sha in record['source_evidence_sha256'].items():
        if name in mapping:
            location = (root/mapping[name]['snapshot']).resolve()
            if not location.is_relative_to(prepared.resolve()) or mapping[name]['sha256'] != sha:
                raise RuntimeError('Mutable aggregate snapshot escapes the prepared evidence')
        else: location = root/name
        if digest(location) != sha: raise RuntimeError('Prepared input bytes changed: '+name)
    for name, sha in record['source_sha256'].items():
        if digest(root/name) != sha: raise RuntimeError('Prepared table source code changed: '+name)
    return record
