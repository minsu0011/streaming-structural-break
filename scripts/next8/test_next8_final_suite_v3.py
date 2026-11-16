"""Final test suite v3: actual legacy threads and isolated synthetic output roots."""
from pathlib import Path
import sys
import os
import subprocess
import tempfile
import uuid
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.next8
from src.next8.common import BASE, read_json, digest, atomic, utc_now, activity, note
from src.next8.runtime_accounting import audit_admitted


def main():
    admission = read_json(BASE / 'runtime/FINAL_AUDIT_ADMISSION.json')
    hb = read_json(BASE / 'runtime/HEARTBEAT.json')
    assert audit_admitted(admission, hb['productive_seconds'], digest(BASE / 'SESSION_NEXT8.json'), 36000.)
    closure = read_json(BASE / 'runtime/FINAL_SOURCE_CLOSURE_NEXT8.json')
    for name, sha in closure['source_sha256'].items():
        assert digest(ROOT / name) == sha, name
    out = BASE / 'final_tests'
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise RuntimeError('Previous final test evidence retained; explicit repair needed for rerun')
    env = dict(os.environ)
    env['NEXT8_TEST_RECEIPT'] = 'FINAL_COMPLETE_TEST_ACCESS_NEXT8.json'
    assert not env.get('NEXT5_TEST_AUDIT_DIRECTORY')
    temporary_root = Path(tempfile.gettempdir()).resolve()
    test_temp = temporary_root / ('next8_final_' + BASE.name + '_' + uuid.uuid4().hex)
    assert test_temp.parent == temporary_root and not test_temp.exists()
    # A fresh system-temp fixture directory is outside protected project data.
    # No pre-existing directory is passed to pytest for recursive replacement.
    command = [sys.executable, '-B', str(ROOT / 'scripts/next8/launch_next8_guarded_tests_v3.py'), '-q', '-p', 'no:cacheprovider', '-p', 'src.next8.test_guard_v2',
               'tests', '-W', 'error::RuntimeWarning', '--junitxml=' + str(out / 'COMPLETE_SUITE.xml'),
               '--basetemp=' + str(test_temp)]
    with activity('Complete existing and NEXT8 test suite', 'FINAL_AUDIT', 'B7_NORMALIZATION_AND_LINEAGE'):
        atomic(out / 'START.json', {'status': 'FROZEN', 'timestamp': utc_now(), 'command': command, 'fresh_synthetic_fixture_directory': str(test_temp),
               'source_closure_sha256': digest(BASE / 'runtime/FINAL_SOURCE_CLOSURE_NEXT8.json')})
        with (out / 'COMPLETE_SUITE.log').open('x', encoding='utf-8') as log:
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        assert process.returncode == 0, 'Final test failure retained for explicit software/scientific classification'
        root = ET.parse(out / 'COMPLETE_SUITE.xml').getroot()
        suites = [root] if root.tag == 'testsuite' else list(root.findall('testsuite'))
        counts = {name: sum(int(s.get(name, 0)) for s in suites) for name in ('tests', 'errors', 'failures', 'skipped')}
        assert counts['tests'] > 0 and counts['errors'] == counts['failures'] == 0
        receipt = BASE / 'audit/FINAL_COMPLETE_TEST_ACCESS_NEXT8.json'
        access = read_json(receipt)
        assert access['status'] == 'PASS' and not access['blocked_reads']
        atomic(out / 'DONE.json', {'status': 'COMPLETE', 'timestamp': utc_now(), 'counts': counts,
               'passed': counts['tests'] - counts['skipped'], 'guard_receipt_sha256': digest(receipt),
               'files_sha256': {name: digest(out / name) for name in ('START.json', 'COMPLETE_SUITE.xml', 'COMPLETE_SUITE.log')},
               'existing_and_next8_tests_included': True, 'no_old_test_receipt_destinations_used': True})
    note('Complete final test suite PASS', str(counts), 'B7_NORMALIZATION_AND_LINEAGE', phase='FINAL_AUDIT')


if __name__ == '__main__':
    main()
