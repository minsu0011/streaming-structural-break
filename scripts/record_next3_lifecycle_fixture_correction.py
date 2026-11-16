"""Keep full-suite failure evidence and retest actual Windows owned-child boundaries."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, atomic_json, digest, utc_now, progress
from src.next3.maintenance import register_job


def main():
    out = BASE/'maintenance/windows_child_tree_fixture_fix'
    atomic_json(out/'REPAIR.json', {'status': 'FIXTURE_CORRECTED_PENDING_RETEST', 'created_utc': utc_now(),
        'finding': 'The Windows virtual-environment Python entry has two owned descendant processes. The supervisor correctly terminated its owned tree, but the test incorrectly expected only the top Popen PID.',
        'fix': 'Capture the actual owned descendant PID set after the child readiness handshake and compare the complete hang receipt set; exclude the test parent explicitly.',
        'failed_full_suite_tests': 1, 'passed_full_suite_tests': 684,
        'failed_full_suite_xml_sha256': digest(BASE/'maintenance/FULL_TESTS_CHECKPOINT_4.xml'),
        'original_test_sha256': digest(out/'original_test_next3_supervisor_lifecycle.py'),
        'corrected_test_sha256': digest(ROOT/'tests/test_next3_supervisor_lifecycle.py'),
        'production_hang_recovery_changed': False, 'scientific_changes': False,
    }, immutable=True)
    register_job({'id': 'F413_LIFECYCLE_RETEST', 'kind': 'maintenance',
        'argv': ['-m', 'pytest', '-q', 'tests/test_next3_supervisor_lifecycle.py', '--junitxml=artifacts/next3/maintenance/SUPERVISOR_LIFECYCLE_TESTS.xml'],
        'description': 'Corrected actual Windows descendant-tree expectation; same lifecycle and clock/resource checks', 'retry_safe': False})
    register_job({'id': 'F414_FULL_TESTS_CORRECTED', 'kind': 'maintenance',
        'argv': ['-m', 'pytest', '-q', '--junitxml=artifacts/next3/maintenance/FULL_TESTS_CHECKPOINT_5.xml'],
        'description': 'Complete suite after the single Windows test-fixture correction; original 684/685 result preserved',
        'requires': ['artifacts/next3/maintenance/SUPERVISOR_LIFECYCLE_TESTS.xml'], 'retry_safe': False})
    register_job({'id': 'F415_PORTABLE_GOLDEN', 'kind': 'maintenance',
        'argv': ['scripts/prepare_next3_portable_runtime_golden.py'],
        'description': 'Raw-free deterministic public-API golden outputs from both frozen references and exact selected lean runtime',
        'requires': ['artifacts/next3/maintenance/FULL_TESTS_CHECKPOINT_5.xml'], 'retry_safe': False})
    register_job({'id': 'F416_PLANNED_PAYLOAD_INVENTORY', 'kind': 'maintenance',
        'argv': ['scripts/package_next3_handoff.py', '--inventory'],
        'description': 'Check file count and size feasibility without creating a final archive or early audit',
        'requires': ['PORTABLE_RUNTIME_GOLDEN_NEXT3.json'], 'retry_safe': False})
    progress('Windows lifecycle fixture correction recorded', 'One failed assertion preserved; actual owned process-tree expectation corrected; no production hang change')


if __name__ == '__main__': main()
