"""Window-expiry contrasts using an availability-defined fixed synthetic subgroup."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[key] = '1'
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.next2.outlier_persistence import aligned_pair_predictions, paired_mean_interval
from src.next2.counterexamples import policy
from src.validation.group_bootstrap import draw_group_multiplicities
from src.next2.session import checkpoint
from src.utils.artifacts import write_json, sha256, utc_now


def main():
    out = ROOT/'artifacts/next2/reused_outlier_age_expiry'; out.mkdir(parents=True, exist_ok=True)
    lock = policy(); n = lock['evaluation_series_per_scenario']
    schedule = np.asarray([lock['tau_cycle'][i % 3] for i in range(n)])
    selected = np.flatnonzero(schedule <= 256); assert len(selected) == 342
    identity = {'script_sha256': sha256(Path(__file__)), 'protocol_sha256': sha256(ROOT/'docs/NEXT2_REUSED_OUTLIER_EXPIRY_PROTOCOL.md'),
        'original_suite_sha256': sha256(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json'), 'source_sha256': sha256(ROOT/'src/next2/outlier_persistence.py'),
        'prior_persistence_receipt_sha256': sha256(ROOT/'artifacts/next2/outlier_persistence/RESULTS.json'),
        'seed': 2026090967, 'replicates': 2000, 'selected_onsets': [128, 256], 'selected_pairs': len(selected), 'posthoc_descriptive': True}
    if (out/'POLICY.json').exists(): assert json.loads((out/'POLICY.json').read_text()) == identity
    else: write_json(out/'POLICY.json', identity)
    _, counts = draw_group_multiplicities(np.arange(len(selected)), 2000, identity['seed']); counts = counts[1:].astype(float)
    models = []
    for eid in ('B101_LONG_VARIANCE_NO_MEMORY', 'B102_LONG_VARIANCE_NO_BAYES', 'V007_VARIANCE_FORECAST'):
        base = ROOT/'artifacts/next2/reused_component_stress_cv0'/eid
        models.append((eid, 'original_CV_fold_0', base/'counterexample/O1', 'prediction.npy'))
    assert len(models) == 3
    curves = []; checkpoints = []; contrasts = []; sources = {}
    for eid, fitted, folder, filename in models:
        receipt = folder/'RESULTS.json'; r = json.loads(receipt.read_text()); assert r['status'] == 'PASS'
        for name in (filename, 'synthetic_tau.npy'): assert sha256(folder/name) == r['files_sha256'][name]
        assert np.all(np.load(folder/'synthetic_tau.npy', allow_pickle=False) == -1)
        p = np.load(folder/filename, allow_pickle=False)
        control, intervention = aligned_pair_predictions(p[np.r_[selected, n+selected]], schedule[selected], 768)
        delta = intervention[:, 1:].astype(float)-control[:, 1:]
        common = {'candidate': eid, 'fitted_model': fitted, 'seed_pairs': len(selected), 'onset_subgroup': '128_or_256'}
        for offset, value in enumerate(delta.mean(axis=0)):
            curves.append({**common, 'post_injection_offset': offset, 'paired_probability_delta': float(value)})
        for offset in (127, 128, 255, 256, 511, 512, 639, 767):
            checkpoints.append({**common, 'post_injection_offset': offset, **paired_mean_interval(delta[:, offset], counts)})
        for boundary in (128, 256, 512):
            contrasts.append({**common, 'boundary': boundary, 'contrast': 'paired_probability_delta_at_boundary_minus_previous_point',
                **paired_mean_interval(delta[:, boundary]-delta[:, boundary-1], counts)})
        sources[receipt.relative_to(ROOT).as_posix()] = sha256(receipt)
    files = {'MEAN_CURVES.csv': curves, 'OFFSET_INTERVALS.csv': checkpoints, 'BOUNDARY_CONTRASTS.csv': contrasts}
    for name, rows in files.items(): pd.DataFrame(rows).to_csv(out/name, index=False)
    write_json(out/'RESULTS.json', {'status': 'PASS', 'identity': identity, 'created_utc': utc_now(), 'models': len(models),
        'input_receipts_sha256': sources, 'files_sha256': {name: sha256(out/name) for name in files}, 'new_fits': 0, 'seal_rows': 0,
        'limitations': 'Posthoc diagnostic restricted by pre-existing observation availability. Conditional fixed-model paired bootstrap; no selection adjustment and no single-feature causal identification. O1 remains a null scenario.'})
    checkpoint('Outlier persistence across fixed age-window expiry boundaries', ['Compare old memory and age evidence', 'Keep all real-DEV gates fixed'])
    print('NEXT2 REUSED OUTLIER AGE EXPIRY PASS', len(models), len(selected), flush=True)


if __name__ == '__main__': main()
