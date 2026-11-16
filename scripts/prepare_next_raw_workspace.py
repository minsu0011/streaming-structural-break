"""Create a detached source checkout with copied raw inputs and hash references.

This is an internal reproducibility workspace, never an early research report.
No existing checkout, artifact, model, raw file, or environment is replaced.
"""
from pathlib import Path
import argparse
import importlib.metadata
import json
import shutil
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.guard import SealGuard
from src.next.data import RAW_SHA
from src.utils.artifacts import write_json, sha256, utc_now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--candidates', nargs='+', required=True)
    args = parser.parse_args()
    if not args.tag.replace('_', '').isalnum():
        raise ValueError('Simple unique workspace tag required')
    guard = SealGuard(ROOT)
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', 'src', 'scripts', 'requirements-lock.txt'], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    destination = ROOT/'artifacts/next/reproduction_workspaces'/args.tag
    if destination.exists():
        raise RuntimeError('Use the existing independent workspace or choose a new explicit tag; never overwrite it')
    request = {'status': 'LOCKED', 'created_utc': utc_now(), 'source_commit': commit,
        'split_sha256': guard.split.digest, 'seal_lock_sha256': guard.lock_sha,
        'numerical_versions': {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'scikit-learn', 'lightgbm', 'numba', 'llvmlite', 'pandas', 'pyarrow', 'joblib', 'threadpoolctl')},
        'candidates': [], 'expected_outcomes_authority': 'Previously frozen original OOF checksums; values are not used as training inputs or to choose a different recipe.',
        'new_quality_search': False}
    for eid in args.candidates:
        path = ROOT/'artifacts/next/experiments'/eid/'RESULTS.json'
        result = json.loads(path.read_text(encoding='utf-8'))
        if result['status'] not in ('ROBUSTNESS_PENDING', 'NEW_DEV_CANDIDATE'):
            raise RuntimeError('Independent raw reproduction requires a valid primary-gate-passing candidate')
        candidate = result['candidate']
        if candidate.get('extension', {}).get('kind') not in ('conditional_variance', 'log_energy', 'serial_variance') or candidate.get('ranking_objective'):
            raise RuntimeError('This raw reconstruction worker supports only the reviewed binary variance and registered evidence families')
        item = {'candidate': candidate, 'original_report_sha256': sha256(path), 'original_folds': {}}
        for fold in range(5):
            folder = path.parent/'full'
            record = json.loads((folder/f'fold_{fold}.json').read_text(encoding='utf-8'))
            for name, digest in record['files_sha256'].items():
                if sha256(folder/name) != digest:
                    raise RuntimeError('Original reference model, predictions or coordinates changed')
            item['original_folds'][str(fold)] = {'ts_auc': result['full']['fold_scores'][fold],
                'prediction_sha256': sha256(folder/f'fold_{fold}.npy'), 'model_sha256': sha256(folder/f'fold_{fold}.joblib')}
        request['candidates'].append(item)
    subprocess.run(['git', 'worktree', 'add', '--detach', str(destination), commit], cwd=ROOT, check=True)
    copied = {}
    paths = ['SEAL_PARTITION_500x4_LOCK.json', 'REDUCED_TEST_USAGE_LEDGER.json',
        'artifacts/validation/SPLIT_LOCK_REAL.json', 'artifacts/validation/GROUP_ASSIGNMENTS.parquet']
    paths.extend(entry['ids_file'] for entry in guard.lock['partitions'].values())
    paths.extend('data/raw/'+name for name in RAW_SHA)
    for relative in paths:
        source, target = ROOT/relative, destination/relative
        if destination.resolve() not in target.resolve().parents:
            raise RuntimeError('Copied reproduction path escapes the new workspace')
        target.parent.mkdir(parents=True, exist_ok=True)
        # Existing tracked metadata is replaced by the already frozen current
        # metadata inside this new checkout only; the original remains intact.
        shutil.copy2(source, target)
        copied[relative] = sha256(source)
        if sha256(target) != copied[relative]:
            raise RuntimeError('Independent input copy did not preserve exact bytes')
    request_path = destination/'configs/next_raw_reproduction_request.json'
    write_json(request_path, request)
    marker = {'status': 'LOCKED', 'created_utc': utc_now(), 'origin_project': str(ROOT),
        'independent_workspace': str(destination), 'canonical_raw': str(ROOT/'data/raw'),
        'source_commit': commit, 'request_sha256': sha256(request_path), 'copied_input_sha256': copied,
        'expected_interpreter': str(ROOT/'.repro-venv/Scripts/python.exe'),
        'data_strategy': 'Exact byte copies of only canonical X_train and y_train_index; no junctions, no source model or OOF arrays, no original feature caches.',
        'environment': 'The separately installed pinned .repro-venv, created for the preceding independent-reproduction stage, is reused. It is separate from the research .venv; this is not a claim that a third environment was freshly installed.',
        'handoff_status': 'Internal research reproduction workspace; not a final handoff or research completion.'}
    write_json(destination/'INDEPENDENT_NEXT_WORKSPACE.json', marker)
    receipt = ROOT/'artifacts/next/independent_raw_workspaces'
    receipt.mkdir(parents=True, exist_ok=True)
    write_json(receipt/(args.tag+'.json'), {**marker, 'marker_sha256': sha256(destination/'INDEPENDENT_NEXT_WORKSPACE.json'),
        'worker_script_sha256': sha256(destination/'scripts/reproduce_next_raw_cv.py')})
    print('PREPARED INDEPENDENT RAW WORKSPACE', destination, flush=True)


if __name__ == '__main__':
    main()
