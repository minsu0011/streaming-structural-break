"""Run the complete unchanged legacy and NEXT7 suites after audit admission."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import uuid
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.next7.common import BASE, save, read_json, digest, utc_now, activity, note
from src.next7.worker_environment import environment


def main():
    from scripts.next7.benchmark_runtime import audit_gate
    audit_gate()
    token = uuid.uuid4().hex
    relative = 'audit/full_regression_' + token
    directory = BASE / relative
    directory.mkdir(parents=True, exist_ok=False)
    # A fresh system temporary directory keeps synthetic test fixtures outside
    # the canonical data/artifact tree denied by the original reader guard.
    temporary = Path(tempfile.gettempdir()) / ('StructuralBreak_NEXT7_full_tests_' + token)
    assert not temporary.exists()
    assert not temporary.resolve().is_relative_to(ROOT.resolve())
    env = environment()
    env.pop('NEXT5_TEST_AUDIT_DIRECTORY', None)
    env['NEXT7_TEST_RECEIPT'] = 'FINAL_FULL_TEST_ACCESS_' + token + '.json'
    env['NUMBA_CACHE_DIR'] = str(BASE / 'numba_cache')
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    command = [sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
        '-p', 'src.next7.test_guard', 'tests', 'src/next7/tests',
        '--basetemp=' + str(temporary), '--junitxml=' + str(directory / 'RESULTS.xml')]
    test_files = sorted((ROOT / 'tests').glob('test_*.py')) + sorted((ROOT / 'src/next7/tests').glob('test_*.py'))
    before = {path.relative_to(ROOT).as_posix(): digest(path) for path in test_files}
    save(relative + '/START.json', {'status': 'STARTED', 'created_utc': utc_now(),
        'command': command, 'working_directory': str(ROOT), 'source_sha256': digest(__file__),
        'test_sources_sha256': before, 'synthetic_temporary_directory': str(temporary),
        'old_test_sources_unchanged': True, 'old_receipt_environment_removed': True,
        'audit_admission_sha256': digest(BASE / 'runtime/FINAL_AUDIT_ADMISSION.json')})
    note('Complete final regression started', 'Original tests plus every NEXT7 test; guarded readers and fresh synthetic temporary directory', 'FINAL_AUDIT', phase='FINAL_AUDIT')
    with activity('Complete legacy and NEXT7 final regression', 'FINAL_AUDIT', 'FINAL_AUDIT'):
        with (directory / 'OUTPUT.log').open('x', encoding='utf-8') as output:
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
        xml = ET.parse(directory / 'RESULTS.xml')
        tests = xml.findall('.//testcase')
        counts = {'tests': len(tests), 'failures': len(xml.findall('.//failure')),
                  'errors': len(xml.findall('.//error')), 'skipped': len(xml.findall('.//skipped'))}
        guard_path = BASE / 'audit' / env['NEXT7_TEST_RECEIPT']
        guarded = read_json(guard_path)
        after = {path.relative_to(ROOT).as_posix(): digest(path) for path in test_files}
        passed = process.returncode == 0 and counts['tests'] > 0 and counts['failures'] == counts['errors'] == 0
        passed = passed and guarded['status'] == 'PASS' and before == after
        result = {'status': 'PASS' if passed else 'FAIL', 'created_utc': utc_now(),
            'returncode': process.returncode, **counts, 'test_sources_unchanged': before == after,
            'guard_receipt': guard_path.relative_to(BASE).as_posix(), 'guard_sha256': digest(guard_path),
            'xml_sha256': digest(directory / 'RESULTS.xml'), 'log_sha256': digest(directory / 'OUTPUT.log'),
            'run_directory': relative, 'test_sources_sha256': before,
            'skips': [{'class': item.get('classname'), 'name': item.get('name'),
                       'reason': item.find('skipped').get('message')} for item in tests if item.find('skipped') is not None],
            'scope': 'Complete test collection with existing test source unchanged. Expected denial fixtures are unit tests, not hidden-data reads.'}
        save(relative + '/DONE.json', result)
        save('audit/FINAL_REGRESSION_LATEST.json', result, False)
        assert passed, 'Complete regression failed; retain receipt and repair before finalization'
    note('Complete final regression passed', str(counts), 'FINAL_AUDIT', phase='FINAL_AUDIT')


if __name__ == '__main__':
    main()
