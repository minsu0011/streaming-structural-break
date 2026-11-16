"""Freeze the long-horizon null study before generating any extended streams."""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.fast_variance_initialization import source_hash as fast_hash
from src.models.cross_order import CrossOrderBlendBundle
from src.streaming.prepared import source_hashes
from src.utils.artifacts import write_json, sha256, utc_now

if __name__ == '__main__':
    out = ROOT/'configs/next_long_null_policy.json'
    if out.exists():
        raise RuntimeError('Long-null policy is immutable')
    original_path = ROOT/'configs/next_synthetic_stress_policy.json'
    original = json.loads(original_path.read_text(encoding='utf-8'))
    models = {'PRIMARY': original['models']['PRIMARY']}
    for eid in ('V007_VARIANCE_FORECAST', 'Q005_VARIANCE_NO_MEMORY', 'Y001_LONG_VARIANCE_ARCH1'):
        models[eid] = f'artifacts/next/experiments/{eid}/full/fold_0.joblib'
    primary = CrossOrderBlendBundle.load(ROOT/models['PRIMARY'])
    files = ['scripts/stress_next_long_null.py', 'src/next/synthetic_variance_stress.py',
        'src/next/alarm_calibration.py', 'src/validation/group_bootstrap.py', 'requirements-lock.txt']
    write_json(out, {'status': 'LOCKED', 'created_utc': utc_now(),
        'parent_policy_sha256': sha256(original_path), 'models': models,
        'model_sha256': {eid: sha256(ROOT/path) for eid, path in models.items()},
        'primary_runtime_receipt': source_hashes(primary, fast_initialization=True),
        'variance_runtime_sha256': fast_hash(), 'source_files_sha256': {name: sha256(ROOT/name) for name in files},
        'evaluation_partition': 2, 'online_points': 16384, 'evaluation_series_per_regime': 1000,
        'report_horizons': [128, 512, 1024, 4096, 16384], 'alphas': [.01, .05],
        'bootstrap_replicates': 2000, 'bootstrap_seed': 202609094,
        'hypothesis': 'Persistent maxima and longer tau windows may improve fixed-horizon AUC while increasing cumulative false alarms when a stream runs much longer than calibration sequences.',
        'calibration': 'Reuse all PRIMARY/V007 calibration predictions and their fixed 5000-null-series maxima thresholds from partition 0. Evaluate Q005/Y001 on exactly the same H/O calibration sequences; their thresholds are locked before partition 2 begins.',
        'long_evaluation': 'Five unchanged AR/ARCH/Student regimes; 1000 new stationary 16384-point futures per regime, generated with independent SeedSequence partition 2. All four models see identical H and O for each index.',
        'generator_precision': 'The original generator draws arrays using a declared simulation horizon. Different requested horizons do not preserve the same Student-noise H prefix; the new partition is a distributional horizon test, not a same-seed cross-horizon replay claim.',
        'online_contract': 'Only H is supplied at initialization; each scalar point is consumed in chronological order. Reporting horizons and maxima are computed outside the detector after causal scores exist.',
        'model_and_threshold_changes': False, 'new_long_scalar_predictions': 327680000,
        'new_calibration_scalar_predictions': 10240000, 'existing_calibration_recomputed': False,
        'runtime_admission': 'All three NEXT models must have exact full 8000-series, 4014405-point production-runtime parity receipts before this script starts.',
        'authority': 'Synthetic null horizon-sensitivity and fixed-state diagnostic only; no selection gate, training objective, raw competition score or production threshold changes.'})
    print('FROZEN LONG NULL HORIZON STUDY', sha256(out), flush=True)
