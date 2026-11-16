"""Freeze the actual-artifact latency protocol before any such measurement."""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.artifacts import write_json, sha256, utc_now


if __name__ == '__main__':
    path = ROOT/'configs/next_all_dev_latency_policy.json'
    if path.exists():
        raise RuntimeError('Retain the existing actual-artifact measurement policy')
    inherited_path = ROOT/'configs/next_latency_policy.json'
    inherited = json.loads(inherited_path.read_text(encoding='utf-8'))
    baseline = ROOT/'artifacts/deployment/final/PRIMARY_DEV_CANDIDATE'
    write_json(path, {'status': 'LOCKED', 'created_utc': utc_now(),
        'inherited_policy_sha256': sha256(inherited_path), 'series_ids': inherited['series_ids'],
        'continuous_points': inherited['continuous_points'], 'block_points': inherited['block_points'],
        'measured_checkpoint_points': inherited['measured_checkpoint_points'],
        'baseline_model_directory': str(baseline.relative_to(ROOT)),
        'baseline_model_sha256': sha256(baseline/'model.joblib'),
        'baseline_descriptor_sha256': sha256(baseline/'model.json'),
        'measurement': 'One fresh-process measurement per distinct all-DEV model/runtime hash; no best-of-repeat selection. Same 128 historical/online pairs, same timers and 10M repeated-buffer stream as the existing CV protocol.',
        'isolation': 'No other Python worker whose cwd is the project or a descendant, except process ancestors. Check before measurement and each one-million-point checkpoint. Ordinary OS background load is not controlled.',
        'expected_values': 'Untimed scalar bundle state plus original compact learner, same all-DEV model; compare every real sampled prediction bitwise. These are training-set fingerprints, not OOF or performance scores.',
        'future_selection': 'Only the already frozen final-role and promotion policies govern eligibility. Measuring a shortlisted artifact does not assign it a final role.',
        'source_sha256': sha256(ROOT/'scripts/benchmark_next_all_dev.py'),
        'runtime_changes_require': 'Explicit revision, unchanged-model full parity, new identity, and a documented reason; never retain the fastest of identical repeats.',
        'seal_rows': 0, 'reduced_new_uses': 0})
    print('ALL-DEV LATENCY POLICY LOCKED', sha256(path), flush=True)
