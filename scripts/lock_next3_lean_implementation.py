"""Freeze E003 source before its first candidate numerical or cost result."""
from pathlib import Path
import ast
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, read_json, atomic_json, digest, utc_now, progress
from src.next3.maintenance import register_job


def main():
    paths = ['src/next3/lean_model.py', 'src/next3/lean_numeric.py', 'src/next3/lean_shared_runtime.py',
        'src/next3/lean_inference.py', 'next3_lean_performance_main.py', 'src/next3/packed_trees.py',
        'configs/next3/LEAN_FEATURE_CONTRACT.json', 'configs/next3/LEAN_NUMERIC_BUILD.json',
        'configs/next3/LEAN_RUNTIME_INPUTS.json', 'docs/NEXT3_LEAN_RUNTIME_PROTOCOL.md',
        'scripts/verify_next3_lean_shared.py', 'scripts/verify_next3_lean_public_api.py',
        'scripts/benchmark_next3_lean_direct.py', 'scripts/benchmark_next3_lean_cold.py',
        'scripts/probe_next3_lean_cold_process.py', 'scripts/materialize_next3_lean_validation.py',
        'tests/test_next3_lean_runtime.py']
    for path in paths:
        if path.endswith('.py'): ast.parse((ROOT/path).read_text(encoding='utf-8'))
    from src.next3.lean_shared_runtime import source_hash
    hypothesis = BASE/'hypotheses/E003_LEAN_SHARED.json'
    atomic_json(BASE/'engineering/lean_shared/IMPLEMENTATION_LOCK.json', {
        'created_utc': utc_now(), 'runtime_sha256': source_hash(), 'hypothesis_sha256': digest(hypothesis),
        'source_sha256': {p: digest(ROOT/p) for p in paths},
        'candidate_prediction_measurements_so_far': 0, 'candidate_cost_measurements_so_far': 0,
        'scientific_fit_count': 0, 'scientific_feature_or_tree_changes': False,
    }, immutable=True)
    register_job({'id': 'E301_LEAN_UNIT', 'kind': 'maintenance',
        'argv': ['-m', 'pytest', '-q', 'tests/test_next3_lean_runtime.py', '--junitxml=artifacts/next3/maintenance/LEAN_UNIT_TESTS.xml'],
        'description': 'Locked E003 malformed-codec, exact extraction, packed-array, streaming and numeric-state tests', 'retry_safe': False})
    register_job({'id': 'E302_LEAN_VALIDATION_ARTIFACT', 'kind': 'maintenance',
        'argv': ['scripts/materialize_next3_lean_validation.py'], 'description': 'Unchanged-byte validation-only lean artifact after exactly 21 passing tests',
        'requires': ['artifacts/next3/maintenance/LEAN_UNIT_TESTS.xml'], 'retry_safe': False})
    register_job({'id': 'E303_LEAN_COLD', 'kind': 'maintenance',
        'argv': ['scripts/benchmark_next3_lean_cold.py'], 'description': 'Six isolated fresh cold subprocesses, fixed RSS/startup cost gates',
        'requires': ['artifacts/next3/canonical_deployment/PERFORMANCE_lean_shared_v1/ASSEMBLY.json'], 'retry_safe': False})
    for number, stage in ((304, 'all_dev'), (305, 'oof'), (306, 'alternate')):
        register_job({'id': f'E{number}_LEAN_{stage.upper()}', 'kind': 'maintenance',
            'argv': ['scripts/verify_next3_lean_shared.py', '--stage', stage],
            'description': 'Every-point exact E003 streaming parity, no new scientific scores or fits',
            'requires': ['artifacts/next3/engineering/lean_shared/cold/COST_GATE_PASS.json'], 'retry_safe': False})
    register_job({'id': 'E307_LEAN_PUBLIC_API', 'kind': 'maintenance',
        'argv': ['scripts/verify_next3_lean_public_api.py'], 'description': 'Actual lean public API every DEV point and reversed reset under write/network denial',
        'requires': ['artifacts/next3/engineering/lean_shared/alternate/RESULTS.json'], 'retry_safe': False})
    register_job({'id': 'E308_LEAN_DIRECT', 'kind': 'maintenance',
        'argv': ['scripts/benchmark_next3_lean_direct.py'], 'description': 'Three fixed paired actual 10M trials versus selected E002; unchanged numeric state',
        'requires': ['artifacts/next3/lean_public_api_all_dev/RESULTS.json'], 'retry_safe': False})
    progress('E003 implementation prospectively frozen', 'Unit checks, six cold processes, complete parity and six direct 10M trials registered', experiment='E003_LEAN_SHARED')


if __name__ == '__main__': main()
