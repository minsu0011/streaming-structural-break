"""Assemble a research artifact only after the frozen engineering criteria pass."""
from pathlib import Path
import sys
import json
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.next2.canonical_model import load
from src.next2.shared_inference import write_artifact
from src.next2.shared_blend_runtime import source_hash
from src.utils.artifacts import write_json, sha256, utc_now


def main():
    receipt = ROOT/'artifacts/next2/latency/BL80_SHARED_V1/RESULTS.json'
    r = json.loads(receipt.read_text())
    assert r['status'] == 'PASS' and r['engineering_replacement_eligible'] and all(r['engineering_replacement_criteria'].values())
    assert r['identity']['runtime_sha256'] == source_hash()
    for path, digest in r['identity']['proofs_sha256'].items(): assert sha256(ROOT/path) == digest
    original = ROOT/'artifacts/next2/canonical_deployment/PERFORMANCE_compact_v1/model.sbnext2'
    assert sha256(original) == r['identity']['model_sha256']
    specification = json.loads((ROOT/'configs/next2/DEPLOYMENT_RESEARCH_PERFORMANCE.json').read_text())
    directory = ROOT/'artifacts/next2/canonical_deployment/PERFORMANCE_shared_v1'
    descriptor = write_artifact(load(original), directory, specification,
        origin='EXACT_SHARED_RUNTIME_FROM_FROZEN_CANONICAL_RESEARCH_MODEL')
    assert descriptor['model_sha256'] == sha256(original)
    write_json(directory/'ASSEMBLY.json', {'status': 'PASS', 'created_utc': utc_now(),
        'engineering_receipt_sha256': sha256(receipt), 'original_canonical_model_sha256': sha256(original),
        'shared_descriptor_sha256': sha256(directory/'model.json'), 'new_fits': 0,
        'scientific_model_bytes_unchanged': True, 'final_promotion': False,
        'pending': ['Actual public read-only all-DEV API', 'Separate final fresh-copy raw training audit']})
    print('NEXT2 SHARED RESEARCH DELIVERY ASSEMBLED', descriptor['model_sha256'], flush=True)


if __name__ == '__main__': main()
