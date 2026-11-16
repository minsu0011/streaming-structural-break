"""Reuse the reviewed raw-reader test boundary with NEXT8-only receipts."""
import os
from src.next5 import final_test_compatibility as compatibility
from src.next5 import test_access_guard as original
from src.next8.common import save, utc_now, digest

keep_legacy_scalar_tests_raw_free = compatibility.keep_legacy_scalar_tests_raw_free


def pytest_configure(config):
    if os.environ.get('NEXT5_TEST_AUDIT_DIRECTORY'):
        raise RuntimeError('Old test receipt destination must not be set')
    compatibility.pytest_configure(config)


def pytest_unconfigure(config):
    compatibility.pytest_unconfigure(config)
    name = os.environ.get('NEXT8_TEST_RECEIPT', 'TEST_ACCESS_GUARD_NEXT8.json')
    if '/' in name or '\\' in name:
        raise ValueError('Receipt must be a NEXT8 filename')
    save('audit/' + name, {'timestamp': utc_now(),
        'status': 'PASS' if not original._blocked else 'BLOCKED_READ_ATTEMPTS',
        'blocked_reads': original._blocked, 'instrumented_calls': len(original._calls),
        'metadata_checks': len(compatibility._metadata_checks),
        'metadata_sha256': compatibility.METADATA_SHA,
        'not_an_OS_sandbox': True, 'source_sha256': digest(__file__)})
