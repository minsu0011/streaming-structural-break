"""Reproduce saved reference bootstrap arrays before freezing their byte lock."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import sys
import json
import numpy as np
from numba import set_num_threads
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.guard import SealGuard
from src.next.references import REFERENCES, load_reference_fold
from src.next.bootstrap import PreparedAUC
from src.validation.group_bootstrap import draw_group_multiplicities
from src.next.bootstrap_integrity import load_reference_samples
from src.utils.artifacts import write_json, sha256, utc_now


def main():
    set_num_threads(4)
    out = ROOT/'artifacts/next/reference_bootstrap'
    lockpath = out/'REPRODUCED_INTEGRITY_LOCK.json'
    if lockpath.exists():
        for role in REFERENCES:
            for fold in range(5):
                load_reference_samples(ROOT, role, fold)
        print('REFERENCE BOOTSTRAP INTEGRITY retained PASS')
        return
    guard = SealGuard(ROOT)
    policy = json.loads((ROOT/'ROBUSTNESS_SPLIT_LOCK.json').read_text(encoding='utf-8'))['bootstrap']
    files, sources = {}, {}
    for role in REFERENCES:
        for fold in range(5):
            frame, prediction, _ = load_reference_fold(ROOT, guard, role, fold)
            code, counts = draw_group_multiplicities(frame.dataset_id.to_numpy(), policy['final_replicates'], policy['seed'] + fold)
            actual = PreparedAUC.create(frame.target, prediction, frame.time_online, code, counts.shape[1]).evaluate(counts)
            name = f'{role}_fold_{fold}.npy'
            np.testing.assert_array_equal(actual, np.load(out/name, allow_pickle=False))
            files[name] = sha256(out/name)
            inputpath = ROOT/'data/processed/oof'/REFERENCES[role]/f'fold_{fold}.npy'
            sources[str(inputpath.relative_to(ROOT))] = sha256(inputpath)
            print('BOOTSTRAP INTEGRITY', role, fold, 'EXACT', flush=True)
    write_json(lockpath, {'status': 'PASS', 'created_utc': utc_now(), 'replicates': policy['final_replicates'],
        'policy_sha256': sha256(ROOT/'ROBUSTNESS_SPLIT_LOCK.json'), 'files_sha256': files,
        'reference_oof_sha256': sources, 'all_replicates_reproduced_exactly': True,
        'source_code_sha256': {name: sha256(ROOT/name) for name in ['src/next/bootstrap.py', 'src/validation/group_bootstrap.py', 'src/scoring/ts_auc.py']},
        'seal_rows': 0, 'reduced_new_usage': 0})


if __name__ == '__main__':
    main()
