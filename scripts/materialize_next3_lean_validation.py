"""Create a validation-only unchanged-byte artifact after locked unit verification."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, read_json, atomic_json, digest, utc_now, progress
from src.next3.lean_inference import write_artifact
from src.next3.lean_shared_runtime import source_hash


def main():
    lock_path = BASE/'engineering/lean_shared/IMPLEMENTATION_LOCK_CORRECTION_001.json'; lock = read_json(lock_path)
    assert lock['runtime_sha256'] == source_hash()
    for p, sha in lock['source_sha256'].items(): assert digest(ROOT/p) == sha, p
    tests = BASE/'maintenance/LEAN_UNIT_TESTS_CORRECTION_001.xml'; suites = ET.parse(tests).getroot().findall('.//testsuite')
    assert suites and sum(int(s.attrib['tests']) for s in suites) == 21
    assert all(int(s.attrib.get('failures', 0)) == int(s.attrib.get('errors', 0)) == int(s.attrib.get('skipped', 0)) == 0 for s in suites)
    out = BASE/'canonical_deployment/PERFORMANCE_lean_shared_v1'
    reference = ROOT/'artifacts/next2/canonical_deployment/PERFORMANCE_shared_v1/model.sbnext2'
    specification = read_json(ROOT/'configs/next2/DEPLOYMENT_RESEARCH_PERFORMANCE.json')
    descriptor = write_artifact(reference.read_bytes(), out, specification,
        origin='E003_VALIDATION_ONLY_EXACT_FROZEN_BYTES_NO_FIT')
    atomic_json(out/'ASSEMBLY.json', {'status': 'PASS', 'created_utc': utc_now(),
        'scope': 'Validation-only prototype, not engineering promotion or independent training',
        'public_source_sha256': {p: sha for p, sha in lock['source_sha256'].items()
            if p in ('src/next3/lean_inference.py', 'next3_lean_performance_main.py', 'scripts/verify_next3_lean_public_api.py')},
        'implementation_lock_sha256': digest(lock_path), 'unit_tests_sha256': digest(tests),
        'model_sha256': digest(out/'model.sbnext2'), 'descriptor_sha256': digest(out/'model.json'),
        'new_fits': 0, 'seal_rows': 0}, immutable=True)
    atomic_json(ROOT/'configs/next3/LEAN_RUNTIME_SELECTION.json', {'status': 'VALIDATION_ONLY',
        'variant': 'LEAN_PACKED_SHARED', 'runtime_sha256': source_hash(), 'created_utc': utc_now(),
        'fixed_gates_required': ['cold_cost', 'all_dev', 'oof', 'alternate_H', 'actual_public_api', 'direct_10M', 'fresh_raw_reproduction', 'final_audit'],
        'scientific_role': 'FROZEN_PERFORMANCE', 'scientific_quality_change': False}, immutable=True)
    progress('E003 validation artifact prepared', 'Unchanged model bytes; fixed engineering gates remain pending', experiment='E003_LEAN_SHARED')


if __name__ == '__main__': main()
