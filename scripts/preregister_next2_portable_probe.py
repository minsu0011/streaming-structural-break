"""Build compact expected synthetic fingerprints from already frozen full-suite predictions."""
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.utils.artifacts import write_json, sha256, utc_now


def main():
    lock = json.loads((ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json').read_text())
    rows = []; sources = {}
    roles = [('DISTILLED', 'counterexamples', 'J102_SERIAL_VARIANCE_512', 'DISTILLED_tree_used_v1', 'next2_canonical_distilled_main'),
        ('PERFORMANCE', 'finalist_counterexamples', 'BL2_C205_DEPENDENCE_80', 'PERFORMANCE_shared_v1', 'next2_shared_performance_main')]
    for role, suite, eid, artifact, entry in roles:
        for scenario in lock['scenarios']:
            sid = scenario['id']; folder = ROOT/'artifacts/next2'/suite/'counterexample'/sid
            r = json.loads((folder/'RESULTS.json').read_text()); assert r['status'] == 'PASS'
            path = folder/(eid+'.npy'); assert sha256(path) == r['files_sha256'][path.name]
            sources[(folder/'RESULTS.json').relative_to(ROOT).as_posix()] = sha256(folder/'RESULTS.json')
            values = np.load(path, mmap_mode='r', allow_pickle=False)
            paired = r['paired_null_controls']; assert values.shape == ((1024 if paired else 512), 1024)
            for control in ([True, False] if paired else [False]):
                start = 512 if paired and not control else 0
                expected = hashlib.sha256(np.asarray(values[start:start+8], dtype=np.float32).tobytes()).hexdigest()
                rows.append({'role': role, 'entry': entry, 'artifact': 'artifacts/next2/canonical_deployment/'+artifact,
                    'model_sha256': sha256(ROOT/'artifacts/next2/canonical_deployment'/artifact/'model.sbnext2'),
                    'scenario': sid, 'control': control, 'seed_indices': list(range(8)), 'horizon': 1024,
                    'expected_probability_sha256': expected})
    destination = ROOT/'configs/next2/PORTABLE_SYNTHETIC_PROBE_LOCK.json'
    if destination.exists(): raise RuntimeError('Portable probe is already fixed')
    write_json(destination, {'status': 'FROZEN', 'created_utc': utc_now(), 'rows': rows,
        'counterexample_lock_sha256': sha256(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json'),
        'script_sha256': sha256(Path(__file__)), 'reference_receipts_sha256': sources,
        'scope': 'First eight seeds of every existing original counterexample cell, selected by index only; prediction equivalence without any new quality-score estimate.'})
    print('NEXT2 PORTABLE PROBE FROZEN', len(rows), 'cells', flush=True)


if __name__ == '__main__': main()
