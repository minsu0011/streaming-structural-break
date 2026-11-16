"""Small raw-data-free public-API reproduction suitable for the GPT handoff copy."""
import os
import sys
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[key] = '1'
from pathlib import Path
import argparse
import importlib
import hashlib
import json
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.next.readonly_bootstrap import install
install()
from src.next2.counterexamples import generate
from src.utils.artifacts import write_json, sha256, utc_now


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True); args = parser.parse_args()
    path = ROOT/'configs/next2/PORTABLE_SYNTHETIC_PROBE_LOCK.json'
    lock = json.loads(path.read_text()); assert lock['status'] == 'FROZEN'
    assert sha256(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json') == lock['counterexample_lock_sha256']
    start = time.perf_counter(); records = []
    for cell in lock['rows']:
        folder = ROOT/cell['artifact']; assert sha256(folder/'model.sbnext2') == cell['model_sha256']
        entry = importlib.import_module(cell['entry'])
        def datasets():
            for seed in cell['seed_indices']:
                h, o, _ = generate(cell['scenario'], seed, partition=1, horizon=cell['horizon'], control=cell['control'])
                yield h, iter(o)
        stream = entry.infer(datasets(), folder); assert next(stream) is None
        values = np.fromiter(stream, dtype=np.float32)
        assert len(values) == len(cell['seed_indices'])*cell['horizon']
        assert np.isfinite(values).all() and np.min(values) >= 0 and np.max(values) <= 1
        digest = hashlib.sha256(values.tobytes()).hexdigest(); assert digest == cell['expected_probability_sha256']
        records.append({'role': cell['role'], 'scenario': cell['scenario'], 'control': cell['control'],
            'points': len(values), 'probability_sha256': digest})
    write_json(Path(args.output), {'status': 'PASS', 'created_utc': utc_now(), 'root': str(ROOT),
        'script_sha256': sha256(Path(__file__)), 'probe_lock_sha256': sha256(path), 'records': records,
        'total_points': sum(r['points'] for r in records), 'new_fits': 0, 'raw_competition_files_accessed': 0,
        'quality_scores_computed': 0, 'elapsed_seconds': time.perf_counter()-start,
        'scope': 'Actual selected public inference APIs against earlier fixed-suite synthetic prediction fingerprints; requires no competition raw data.'})
    print('NEXT2 PORTABLE REPLAY PASS', len(records), sum(r['points'] for r in records), flush=True)


if __name__ == '__main__': main()
