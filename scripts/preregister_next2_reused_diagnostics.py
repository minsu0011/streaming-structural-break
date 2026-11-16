"""Complete the three reused-ablation diagnostic omissions without refitting."""
from pathlib import Path
import sys
import json
import argparse
import subprocess
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.utils.artifacts import write_json, sha256, utc_now


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--benchmark', action='store_true'); args = parser.parse_args()
    names = ['B101_LONG_VARIANCE_NO_MEMORY', 'B102_LONG_VARIANCE_NO_BAYES', 'V007_VARIANCE_FORECAST']
    models = []
    for eid in names:
        folder = ROOT/'artifacts/next/experiments'/eid
        result = json.loads((folder/'RESULTS.json').read_text())
        model = folder/'full/fold_0.joblib'
        assert model.exists() and result['full']['status'] == 'COMPLETE' and result['status'] not in ('SCREEN_FAIL', 'BUG')
        models.append({'experiment_id': eid, 'candidate': result['candidate'], 'full_cv_status': result['status'],
            'model_path': model.relative_to(ROOT).as_posix(), 'model_sha256': sha256(model),
            'source_result_sha256': sha256(folder/'RESULTS.json'), 'outer_fold_model': 0})
    stress_path = ROOT/'configs/next2/REUSED_COMPONENT_STRESS_LOCK.json'
    timing_path = ROOT/'configs/next2/REUSED_COMPONENT_MEASUREMENT_LOCK.json'
    timing = json.loads((ROOT/'configs/next2/COMPONENT_MEASUREMENT_LOCK.json').read_text())
    timing.update(models=models, driver_sha256=sha256(ROOT/'scripts/benchmark_next2_reused_components.py'),
        supplement_of_sha256=sha256(ROOT/'configs/next2/COMPONENT_MEASUREMENT_LOCK.json'))
    for path, policy in ((stress_path, {'status': 'FROZEN_DIAGNOSTIC_MENU', 'models': models,
        'counterexample_lock_sha256': sha256(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json'), 'new_fits': 0,
        'protocol_sha256': sha256(ROOT/'docs/NEXT2_REUSED_COMPONENT_DIAGNOSTICS.md'),
        'selection': 'Descriptive supplement for reused component evidence; no candidate promotion'}), (timing_path, timing)):
        if path.exists():
            stored = json.loads(path.read_text()); stored.pop('supplement_created_utc', None); assert stored == policy
        else:
            write_json(path, {**policy, 'supplement_created_utc': utc_now()})
    if args.benchmark:
        for eid in names:
            subprocess.run([sys.executable, 'scripts/benchmark_next2_reused_components.py', '--candidate', eid],
                check=True, cwd=ROOT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    print('NEXT2 REUSED DIAGNOSTIC LOCK COMPLETE', names, flush=True)


if __name__ == '__main__': main()
