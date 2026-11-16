"""Reuse reviewed legacy test guard, writing receipts exclusively under NEXT6."""
import os
from src.next5 import final_test_compatibility as compatibility
from src.next5 import test_access_guard as original
from src.next6.common import save,utc_now,digest,ROOT

keep_legacy_scalar_tests_raw_free=compatibility.keep_legacy_scalar_tests_raw_free

def pytest_configure(config):
    if os.environ.get('NEXT5_TEST_AUDIT_DIRECTORY'):raise RuntimeError('Legacy audit directory must not be set in NEXT6')
    compatibility.pytest_configure(config)

def pytest_unconfigure(config):
    compatibility.pytest_unconfigure(config)
    save('final_audit/TEST_ACCESS_GUARD.json',{'timestamp':utc_now(),'status':'PASS' if not original._blocked else 'BLOCKED_READ_ATTEMPTS',
        'blocked_reads':original._blocked,'instrumented_calls':len(original._calls),'metadata_admission_checks':len(compatibility._metadata_checks),
        'metadata_sha256':compatibility.METADATA_SHA,'legacy_model_loads':compatibility._model_loads,
        'original_test_source_changed':False,'raw_and_old_array_guard_retained':True,'not_an_OS_sandbox':True,
        'compatibility_source_sha256':digest(ROOT/'src/next5/final_test_compatibility.py')})
