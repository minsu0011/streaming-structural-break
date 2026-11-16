"""Bind D005 implementation and schedule it after isolated E003 cost measurements."""
from pathlib import Path
import ast
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, read_json, atomic_json, digest, utc_now, progress
from src.next3.maintenance import register_job


def main():
    hypothesis = BASE/'hypotheses/D005_OUTLIER_STATE_TRANSFER.json'
    paths = ['src/next3/state_transfer.py', 'scripts/forensic_next3_outlier_state_transfer.py',
        'tests/test_next3_state_transfer.py', 'docs/NEXT3_OUTLIER_STATE_TRANSFER_PROTOCOL.md',
        'src/next2/counterexamples.py', 'src/next2/shared_blend_runtime.py',
        'src/validation/group_bootstrap.py']
    for path in paths:
        if path.endswith('.py'): ast.parse((ROOT/path).read_text(encoding='utf-8'))
    atomic_json(BASE/'outlier_state_transfer/IMPLEMENTATION_LOCK.json', {
        'created_utc': utc_now(), 'hypothesis_sha256': digest(hypothesis),
        'source_sha256': {p: digest(ROOT/p) for p in paths}, 'new_fits': 0,
        'intervention_result_measurements_so_far': 0, 'scientific_menu_reopened': False,
    }, immutable=True)
    register_job({'id': 'E309_LEAN_ENGINEERING_DECISION', 'kind': 'maintenance',
        'argv': ['scripts/assess_next3_lean_engineering.py'], 'description': 'Fixed engineering eligibility only after all actual E003 parity/cost proofs',
        'requires': ['artifacts/next3/engineering/lean_shared/direct/RESULTS.json'], 'retry_safe': False})
    register_job({'id': 'F409_D005_STATE_UNITS', 'kind': 'maintenance',
        'argv': ['-m', 'pytest', '-q', 'tests/test_next3_state_transfer.py', '--junitxml=artifacts/next3/maintenance/STATE_TRANSFER_UNIT_TESTS.xml'],
        'description': 'Exact frozen-state clone, non-aliasing and complete two-way transfer checks',
        'requires': ['artifacts/next3/engineering/lean_shared/direct/RESULTS.json'], 'retry_safe': False})
    register_job({'id': 'F412_D005_STATE_CHARACTERIZATION', 'kind': 'maintenance',
        'argv': ['scripts/forensic_next3_outlier_state_transfer.py'], 'description': 'Original O1 cohort matched-state interventions, exact old predictions, no new candidates or fits',
        'requires': ['artifacts/next3/maintenance/STATE_TRANSFER_UNIT_TESTS.xml'], 'retry_safe': False})
    progress('D005 implementation prospectively frozen', 'Seven unit checks then original-cohort characterization after isolated E003 timing')


if __name__ == '__main__': main()
