"""Verified reference-bootstrap artifacts; never silently rebuild on mismatch."""
import json
from pathlib import Path
import numpy as np
from src.utils.artifacts import sha256


def load_reference_samples(root, role, fold):
    root = Path(root)
    out = root/'artifacts/next/reference_bootstrap'
    lock = json.loads((out/'REPRODUCED_INTEGRITY_LOCK.json').read_text(encoding='utf-8'))
    if lock['status'] != 'PASS' or lock['policy_sha256'] != sha256(root/'ROBUSTNESS_SPLIT_LOCK.json'):
        raise RuntimeError('Reference bootstrap integrity lock is invalid')
    for name, digest in lock['source_code_sha256'].items():
        if sha256(root/name) != digest:
            raise RuntimeError('Reference bootstrap scientific code changed')
    name = f'{role}_fold_{fold}.npy'
    if sha256(out/name) != lock['files_sha256'][name]:
        raise RuntimeError('Reference bootstrap replicate bytes changed')
    values = np.load(out/name, allow_pickle=False)
    if values.shape != (lock['replicates'] + 1,) or not np.isfinite(values).all():
        raise RuntimeError('Invalid reference bootstrap replicate array')
    return values
