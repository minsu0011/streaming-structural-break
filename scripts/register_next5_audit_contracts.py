from pathlib import Path
import ast
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next5.artifacts import BASE,atomic_json,digest,utc_now,progress
from src.next5.scheduler import register_job


def main():
    names=('src/next5/test_access_guard.py','scripts/run_next5_full_tests.py','tests/test_next5_test_access_guard.py')
    for name in names:ast.parse((ROOT/name).read_text(encoding='utf-8'),filename=name)
    atomic_json(BASE/'validation/FULL_TEST_ACCESS_POLICY.json',{'created_utc':utc_now(),
        'sources_sha256':{name:digest(ROOT/name) for name in names},
        'scope':'Check compatibility of guarded NumPy/pandas/PyArrow readers with temporary synthetic loader fixtures before the separate final suite.',
        'complete_suite_is_deferred_to_final_audit':True},immutable=True)
    register_job({'id':'TESTGUARD000_COMPATIBILITY','kind':'validation','priority':35,
        'argv':['-m','pytest','-q','tests/test_next5_test_access_guard.py','tests/test_loader.py','tests/test_research_guards.py',
            '-p','src.next5.test_access_guard','--junitxml=artifacts/next5/validation/TEST_ACCESS_GUARD.xml'],
        'description':'Canonical real-data denial remains compatible with synthetic test fixtures; complete suite remains in final audit','retry_safe':False})
    progress('Final-test access contract prepared','Reviewed all test file-read sites; legacy fitted weights and synthetic fixtures allowed, canonical observations and old per-series arrays denied')


if __name__=='__main__':main()
