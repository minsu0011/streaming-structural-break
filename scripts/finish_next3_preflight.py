"""Fail-closed gate before any NEXT3 model experiments."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, atomic_json, read_json, digest, utc_now, progress
from src.next.guard import SealGuard


def main():
    guard = SealGuard(ROOT)
    guard.check_state()
    report = read_json(BASE/'preflight/INITIAL_RESTORE.json')
    xml = BASE/'preflight/GUARD_SCORER_TESTS.xml'
    suites = list(ET.parse(xml).getroot().iter('testsuite'))
    assert suites and all(int(s.attrib.get(k, 0)) == 0 for s in suites for k in ('errors', 'failures', 'skipped'))
    reports = {role: read_json(BASE/'preflight'/role/'RESULTS.json') for role in ('distilled', 'performance')}
    assert all(r['status'] == 'PASS' and r['series'] == 8000 and r['points'] == 4014405 for r in reports.values())
    report.update(status='PASS', created_utc=utc_now(), pending=[],
        fresh_guard_and_scorer_tests=sum(int(s.attrib['tests']) for s in suites),
        test_report_sha256=digest(xml), frozen_prediction_replays=reports,
        application_guard_limitation='Predicate-filtered DEV and Python guards; full-file hashing is not seal observation access; not an OS sandbox.')
    atomic_json(BASE/'preflight/NEXT3_PREFLIGHT.json', report, immutable=True)
    atomic_json(ROOT/'NEXT3_PREFLIGHT.json', report, immutable=True)
    session = read_json(BASE/'SESSION.json')
    session['status'] = 'ACTIVE_RESEARCH'
    atomic_json(BASE/'SESSION.json', session)
    progress('NEXT3 preflight complete', 'All raw/model/guard/scorer/public fingerprint checks pass')
    print('NEXT3 PREFLIGHT PASS', report['fresh_guard_and_scorer_tests'], 'fresh tests; both 4,014,405-point fingerprints exact', flush=True)


if __name__ == '__main__':
    main()
