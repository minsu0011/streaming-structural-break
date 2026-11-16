"""Execute the final source-pinned complete test suite during additional audit."""
from pathlib import Path
import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.research_clock import ResearchClock
from src.utils.artifacts import write_json, sha256, utc_now


def main():
    clock = ResearchClock(ROOT).snapshot()
    if not clock['minimum_research_met'] or not clock['final_audit_started']:
        raise RuntimeError('Final suite execution requires the additional audit phase')
    portable = json.loads((ROOT/'artifacts/next/portable_delivery/RESULTS.json').read_text(encoding='utf-8'))
    environment = json.loads((ROOT/'artifacts/next/portable_delivery/ENVIRONMENT.json').read_text(encoding='utf-8'))
    if portable['status'] != 'PASS' or environment['status'] != 'PASS' or not environment['all_pinned_versions_exact']:
        raise RuntimeError('The newly installed pinned environment must pass portable inference first')
    python = Path(environment['python'])
    if not python.resolve().is_relative_to((ROOT/'.next-portable-v3/environment').resolve()):
        raise RuntimeError('Final suite must use the freshly installed environment')
    out = ROOT/'artifacts/next/final_tests'
    if out.exists():
        raise RuntimeError('Preserve the first final-suite execution; review failures explicitly')
    out.mkdir(parents=True)
    paths = sorted({*ROOT.glob('next*_main.py'), *[path for folder in ('src', 'scripts', 'tests') for path in (ROOT/folder).rglob('*.py')]})
    source_files = {path.relative_to(ROOT).as_posix(): sha256(path) for path in paths}
    identity = {'created_utc': utc_now(), 'python_executable': str(python), 'clock_at_start': clock,
        'fresh_environment_receipt_sha256': sha256(ROOT/'artifacts/next/portable_delivery/ENVIRONMENT.json'),
        'source_files_sha256': source_files, 'requirements_sha256': sha256(ROOT/'requirements-lock.txt')}
    write_json(out/'IDENTITY.json', identity)
    start = time.perf_counter()
    command = [str(python), '-m', 'pytest', '-q', '--junitxml='+str(out/'pytest.xml')]
    with (out/'STDOUT.txt').open('w', encoding='utf-8') as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            print(line.rstrip(), flush=True)
        code = process.wait()
    suite = ET.parse(out/'pytest.xml').getroot()
    tests = suite.findall('.//testcase')
    failures = sum(test.find('failure') is not None for test in tests)
    errors = sum(test.find('error') is not None for test in tests)
    skipped = sum(test.find('skipped') is not None for test in tests)
    current = {relative: sha256(ROOT/relative) for relative in source_files}
    passed = code == 0 and len(tests) >= 350 and failures == errors == skipped == 0 and current == source_files
    write_json(out/'RESULTS.json', {'status': 'PASS' if passed else 'FAIL', 'created_utc': utc_now(),
        'test_cases': len(tests), 'failures': failures, 'errors': errors, 'skipped': skipped, 'exit_code': code,
        'source_files_unchanged_during_suite': current == source_files, 'elapsed_seconds': time.perf_counter()-start,
        'python_executable': str(python), 'fresh_pinned_environment': True,
        'files_sha256': {name: sha256(out/name) for name in ('IDENTITY.json', 'pytest.xml', 'STDOUT.txt')},
        'source_files_sha256': source_files, 'driver_sha256': sha256(Path(__file__))})
    if not passed:
        raise RuntimeError('Final test suite or source consistency failed')
    print('FINAL NEXT TESTS PASS', len(tests), flush=True)


if __name__ == '__main__':
    main()
