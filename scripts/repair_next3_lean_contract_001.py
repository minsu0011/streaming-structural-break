"""Preserve the failed E003 source revision and correct a feature-catalog assertion."""
from pathlib import Path
import shutil
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, read_json, atomic_json, digest, utc_now, progress
from src.next3.maintenance import register_job


def main():
    out = BASE/'engineering/lean_shared'; old_lock = out/'IMPLEMENTATION_LOCK.json'; lock = read_json(old_lock)
    repair = out/'implementation_correction_001'; repair.mkdir(parents=True, exist_ok=False)
    assert not (out/'cold').exists() and not (BASE/'canonical_deployment/PERFORMANCE_lean_shared_v1').exists()
    for relative, expected in lock['source_sha256'].items():
        path = ROOT/relative; assert digest(path) == expected, relative
        target = repair/'original_sources'/relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target); assert digest(target) == expected
    shutil.copyfile(old_lock, repair/'ORIGINAL_IMPLEMENTATION_LOCK.json')
    runtime = ROOT/'src/next3/lean_shared_runtime.py'; source = runtime.read_text(encoding='utf-8')
    before = "if len(old_names) != 51 or len(full_new) != 26: raise ValueError('Frozen 51+26 feature ordering required')"
    after = "if old_names[:51] != primary['names'][:51] or len(full_new) != 26: raise ValueError('Frozen 51+26 feature ordering required')"
    assert before in source; runtime.write_text(source.replace(before, after), encoding='utf-8', newline='\n')
    changed = [runtime.relative_to(ROOT).as_posix()]
    for relative in ('scripts/verify_next3_lean_shared.py', 'scripts/benchmark_next3_lean_cold.py', 'scripts/materialize_next3_lean_validation.py'):
        path = ROOT/relative; source = path.read_text(encoding='utf-8'); assert 'IMPLEMENTATION_LOCK.json' in source
        source = source.replace('IMPLEMENTATION_LOCK.json', 'IMPLEMENTATION_LOCK_CORRECTION_001.json')
        if relative.endswith('materialize_next3_lean_validation.py'):
            source = source.replace('LEAN_UNIT_TESTS.xml', 'LEAN_UNIT_TESTS_CORRECTION_001.xml')
        path.write_text(source, encoding='utf-8', newline='\n'); changed.append(relative)
    from src.next3.lean_shared_runtime import source_hash
    new_lock = out/'IMPLEMENTATION_LOCK_CORRECTION_001.json'
    atomic_json(new_lock, {'created_utc': utc_now(), 'runtime_sha256': source_hash(),
        'hypothesis_sha256': lock['hypothesis_sha256'], 'source_sha256': {p: digest(ROOT/p) for p in lock['source_sha256']},
        'supersedes_lock_sha256': digest(old_lock), 'original_sources_preserved': repair.relative_to(ROOT).as_posix(),
        'scope': 'Catalog guard correction only, before any lean prediction or cost measurement; no arithmetic or scientific model change',
        'candidate_cost_measurements_so_far': 0, 'completed_candidate_prediction_measurements_so_far': 0,
    }, immutable=True)
    atomic_json(repair/'REPAIR.json', {'status': 'CORRECTED_PENDING_RETEST', 'created_utc': utc_now(),
        'finding': 'Legacy feature_names returns 57 total columns, of which the frozen J102 model selects the first 51 ABCD columns. The new guard incorrectly required the complete legacy catalog to have length 51.',
        'fix': 'Require exact names/order for the frozen first 51 columns and exactly 26 serial columns.',
        'failed_unit_tests': 4, 'passed_unit_tests': 17, 'old_unit_xml_sha256': digest(BASE/'maintenance/LEAN_UNIT_TESTS.xml'),
        'validation_assembly_stopped_at_failed_unit_gate': True, 'model_artifacts_created': 0,
        'old_lock_sha256': digest(old_lock), 'new_lock_sha256': digest(new_lock), 'changed_sources': changed,
        'no_scientific_fits_or_quality_scores': True, 'cost_thresholds_and_trial_orders_unchanged': True,
    }, immutable=True)
    register_job({'id': 'E301A_LEAN_UNIT_CORRECTED', 'kind': 'maintenance',
        'argv': ['-m', 'pytest', '-q', 'tests/test_next3_lean_runtime.py', '--junitxml=artifacts/next3/maintenance/LEAN_UNIT_TESTS_CORRECTION_001.xml'],
        'description': 'One documented E003 catalog-guard correction; repeat unchanged 21 tests; original failures preserved', 'retry_safe': False})
    register_job({'id': 'E302A_LEAN_VALIDATION_ARTIFACT_CORRECTED', 'kind': 'maintenance',
        'argv': ['scripts/materialize_next3_lean_validation.py'],
        'description': 'Validation-only unchanged canonical bytes after corrected-unit receipt passes',
        'requires': ['artifacts/next3/maintenance/LEAN_UNIT_TESTS_CORRECTION_001.xml'], 'retry_safe': False})
    progress('E003 feature catalog guard corrected', 'Original source lock and four failed tests preserved; no candidate cost trial yet', experiment='E003_LEAN_SHARED')


if __name__ == '__main__': main()
