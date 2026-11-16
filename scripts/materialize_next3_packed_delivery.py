"""Materialize a provisional exact packed artifact after its registered cost gates."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, read_json, atomic_json, digest, utc_now, progress
from src.next3.packed_inference import write_artifact
from src.next3.packed_shared_runtime import source_hash
from src.next2.canonical_model import load
from scripts.verify_next3_packed_shared import checked_implementation


def main():
    implementation = checked_implementation()
    timing = BASE/'engineering/packed_shared/timing/RESULTS.json'; result = read_json(timing)
    assert result['status'] == 'PASS'; variant = result['provisional_selected_variant']
    if variant is None or not result['variants'][variant]['engineering_gate_pass']:
        raise RuntimeError('Fixed packed engineering gates did not select an eligible variant')
    if (timing.parent/'MEASUREMENT_ACTIVITY_CAVEAT.json').exists():
        raise RuntimeError('Known timing interference prevents accepting these measurements')
    for name, sha in result['files_sha256'].items(): assert digest(timing.parent/name) == sha
    assert digest(timing.parent/'PAIRED_TRIALS.csv') == result['summary_sha256']
    original = ROOT/'artifacts/next2/canonical_deployment/PERFORMANCE_shared_v1/model.sbnext2'
    frozen = read_json(BASE/'preflight/FROZEN_REFERENCES.json')['protected_sha256']
    assert digest(original) == frozen[original.relative_to(ROOT).as_posix()]
    selection_path = ROOT/'configs/next3/PACKED_RUNTIME_SELECTION.json'
    if selection_path.exists(): raise RuntimeError('Preserve an earlier packed runtime selection')
    out = BASE/'canonical_deployment/PERFORMANCE_packed_shared_v1'
    descriptor = write_artifact(load(original), out, read_json(ROOT/'configs/next2/DEPLOYMENT_RESEARCH_PERFORMANCE.json'),
        variant=variant, origin='EXACT_NEXT3_PACKING_OF_UNCHANGED_FROZEN_PERFORMANCE')
    assert descriptor['model_sha256'] == digest(original)
    atomic_json(selection_path, {'status': 'PROVISIONAL_ENGINEERING_ELIGIBLE', 'created_utc': utc_now(),
        'variant': variant, 'runtime_sha256': source_hash(), 'implementation_lock_sha256': implementation,
        'timing_receipt_sha256': digest(timing), 'scientific_role': 'FROZEN_PERFORMANCE',
        'scientific_quality_change': False, 'final_promotion': False}, immutable=True)
    public_sources = ['src/next3/packed_inference.py', 'next3_packed_performance_main.py',
                      'scripts/verify_next3_packed_public_api.py', 'scripts/materialize_next3_packed_delivery.py']
    atomic_json(out/'ASSEMBLY.json', {'status': 'PASS', 'created_utc': utc_now(), 'variant': variant,
        'timing_receipt_sha256': digest(timing), 'selection_sha256': digest(selection_path),
        'model_bytes_identical_to_frozen': True, 'model_sha256': digest(original),
        'public_source_sha256': {name: digest(ROOT/name) for name in public_sources},
        'new_fits': 0, 'final_promotion': False,
        'pending': ['Actual read-only public API on every DEV point', 'Final fresh pinned raw public training reproduction', 'Final audit decision']}, immutable=True)
    progress('Provisional packed delivery assembled', variant+'; exact canonical bytes, final API/reproduction/audit gates pending', experiment='E002_PACKED_SHARED')
    print('NEXT3 PACKED RESEARCH DELIVERY ASSEMBLED', variant, descriptor['model_sha256'], flush=True)


if __name__ == '__main__': main()
