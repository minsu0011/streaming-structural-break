"""Audit retained independent raw payloads and collect small portable receipts.

This performs no fit, prediction, candidate choice, or early final handoff.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.guard import SealGuard
from src.next.progress import checkpoint
from src.utils.artifacts import write_json, sha256, utc_now


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def compare_all_dev(original, independent):
    require(original['status'] == independent['status'] == 'PASS', 'All-DEV fit did not pass')
    fields = ('model_sha256', 'all_dev_prediction_sha256', 'training')
    for field in fields:
        require(original[field] == independent[field], 'Independent all-DEV mismatch: '+field)
    require(original['fingerprint_points'] == independent['all_dev_fingerprint_points'] == 4014405,
            'All-DEV fingerprint coverage differs')
    require(original['training_quality_score_computed'] is False and independent['training_quality_score_computed'] is False,
            'Training-set fingerprint must not become a quality score')
    require(original['training']['training_series'] == 8000 and original['training']['seal_rows_fitted'] == 0,
            'All-DEV training identity is incomplete or includes a seal')
    return {field: original[field] for field in fields if field != 'training'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace-tag', required=True)
    parser.add_argument('--refit-tag', default='research_v1')
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    require(all(value.replace('_', '').isalnum() for value in vars(args).values()), 'Invalid tag')
    out = ROOT/'artifacts/next/independent_raw_validation'/args.tag
    require(not out.exists(), 'Retain completed independent audit; use an explicit new audit tag')
    guard = SealGuard(ROOT)
    receipt_path = ROOT/'artifacts/next/independent_raw_workspaces'/(args.workspace_tag+'.json')
    receipt = read(receipt_path)
    workspace = Path(receipt['independent_workspace']).resolve()
    require(workspace == (ROOT/'artifacts/next/reproduction_workspaces'/args.workspace_tag).resolve(), 'Wrong workspace')
    marker_path = workspace/'INDEPENDENT_NEXT_WORKSPACE.json'
    request_path = workspace/'configs/next_raw_reproduction_request.json'
    require(sha256(marker_path) == receipt['marker_sha256'], 'Workspace marker changed')
    require(sha256(request_path) == receipt['request_sha256'], 'Reproduction request changed')
    require(sha256(workspace/'scripts/reproduce_next_raw_cv.py') == receipt['worker_script_sha256'], 'Worker changed')
    marker, request = read(marker_path), read(request_path)
    require(request['split_sha256'] == guard.split.digest and request['seal_lock_sha256'] == guard.lock_sha, 'Split or seal lock mismatch')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=workspace, text=True).strip()
    require(commit == marker['source_commit'], 'Independent checkout commit changed')
    # Compare exact Git blobs, bypassing the text clean filter. A CRLF blob
    # committed before the current .gitattributes can otherwise appear changed
    # despite byte-identical checkout contents (observed on requirements-lock).
    tree = subprocess.check_output(['git', 'ls-tree', '-r', '--format=%(objectname) %(path)',
        'HEAD', '--', 'src', 'scripts', 'requirements-lock.txt'], cwd=workspace, text=True)
    source_files = {}
    for line in tree.splitlines():
        expected_blob, relative = line.split(' ', 1)
        payload = (workspace/relative).read_bytes()
        actual_blob = hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
        require(actual_blob == expected_blob, 'Independent source bytes differ from commit: '+relative)
        source_files[relative] = hashlib.sha256(payload).hexdigest()
    inputs_verified = {}
    for relative, digest in marker['copied_input_sha256'].items():
        require(sha256(workspace/relative) == digest, 'Copied input changed: '+relative)
        require(sha256(ROOT/relative) == digest, 'Original frozen input changed: '+relative)
        inputs_verified[relative] = digest
    independent_root = workspace/'artifacts/next/raw_reproduction'
    top = read(independent_root/'RESULTS.json')
    require(top['status'] == 'PASS' and top['request_sha256'] == sha256(request_path), 'Incomplete worker receipt')
    require(top['workspace_marker_sha256'] == sha256(marker_path) and top['access_guard']['denied_origin_accesses'] == 0,
            'Original-workspace access guard failure')
    manifests = {}
    for path in (workspace/'data/independent_next').glob('*/MANIFEST.json'):
        manifest = read(path)
        require(manifest['status'] == 'PASS' and manifest['series'] == 8000 and manifest['points'] == 4014405,
                'Incomplete independent raw feature bank')
        for relative, digest in manifest['files_sha256'].items():
            require(sha256(path.parent/relative) == digest, 'Independent raw feature or coordinate bytes changed')
        manifests[sha256(path)] = path
    copied_files = [receipt_path, marker_path, request_path, independent_root/'RESULTS.json',
        workspace/'scripts/reproduce_next_raw_cv.py', Path(__file__)]
    candidates = []
    for item in request['candidates']:
        eid = item['candidate']['experiment_id']
        folder = independent_root/eid
        record = read(folder/'RESULTS.json')
        original_path = ROOT/'artifacts/next/experiments'/eid/'RESULTS.json'
        require(sha256(original_path) == item['original_report_sha256'], 'Original candidate report changed')
        require(record['candidate'] == item['candidate'] == read(original_path)['candidate'], 'Candidate definition changed')
        require(record['status'] == 'PASS' and record['total_oof_points'] == 4014405 and len(record['folds']) == 5,
                'Incomplete independent five-fold reproduction')
        require(Path(record['python_executable']).resolve() == Path(marker['expected_interpreter']).resolve(), 'Wrong reproduction interpreter')
        require(record['numerical_versions'] == request['numerical_versions'], 'Numerical environment differs')
        require(record['fresh_raw_feature_manifest_sha256'] in manifests, 'Missing independent feature bank')
        fold_rows = []
        for fold in range(5):
            path = folder/f'fold_{fold}.json'
            row = read(path)
            expected = item['original_folds'][str(fold)]
            require(row == record['folds'][fold] and row['status'] == 'PASS', 'Fold receipt mismatch')
            require(row['fold'] == fold and row['candidate'] == eid and row['original_reference'] == expected, 'Fold identity mismatch')
            for relative, digest in row['files_sha256'].items():
                require(sha256(folder/relative) == digest, 'Independent fold payload changed')
                require(sha256(original_path.parent/'full'/relative) == digest, 'Original and independent fold payloads differ')
            require(row['model_bytes_equal'] and row['all_float32_predictions_bitwise_equal_by_sha256'] and row['score_equal'],
                    'Independent model or OOF values differ')
            require(abs(row['ts_auc'] - expected['ts_auc']) <= 1e-14, 'Reproduced fold score differs')
            fold_rows.append(row)
            copied_files.append(path)
        refit = ROOT/'artifacts/next/refits'/eid/('cache_'+args.refit_tag)
        original_final, independent_final = read(refit/'RESULTS.json'), read(folder/'ALL_DEV.json')
        comparison = compare_all_dev(original_final, independent_final)
        require(sha256(refit/'model.joblib') == sha256(folder/'all_dev.joblib') == comparison['model_sha256'], 'All-DEV payload mismatch')
        require(sha256(refit/'model.json') == original_final['descriptor_sha256'], 'All-DEV descriptor changed')
        require(record['all_dev'] == independent_final, 'Nested all-DEV receipt differs')
        candidates.append({'candidate': eid, 'status': 'PASS', 'cv_points': 4014405, 'folds': fold_rows,
            'all_dev_match': comparison, 'all_dev_training_fingerprints_equal': True,
            'all_dev_training': original_final['training'], 'original_cache_refit_report_sha256': sha256(refit/'RESULTS.json'),
            'independent_candidate_report_sha256': sha256(folder/'RESULTS.json'),
            'fresh_raw_feature_manifest_sha256': record['fresh_raw_feature_manifest_sha256']})
        copied_files.extend([folder/'RESULTS.json', folder/'ALL_DEV.json', refit/'RESULTS.json'])
    require(len(top['candidates']) == len(candidates) and set(top['candidates']) == {row['candidate'] for row in candidates},
            'Worker candidate coverage mismatch')
    require(top['original_prediction_points_reproduced'] == 4014405*len(candidates), 'Total OOF coverage mismatch')
    out.mkdir(parents=True)
    portable = {}
    for path in [*copied_files, *manifests.values()]:
        relative = ('independent/'+path.relative_to(workspace).as_posix()) if workspace in path.parents else ('origin/'+path.relative_to(ROOT).as_posix())
        target = out/'receipts'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        require(sha256(target) == sha256(path), 'Receipt copy differs')
        portable[target.relative_to(out).as_posix()] = sha256(target)
    result = {'status': 'PASS', 'created_utc': utc_now(), 'source_commit': commit,
        'worker_script_sha256': receipt['worker_script_sha256'], 'collector_sha256': sha256(Path(__file__)),
        'workspace': str(workspace), 'request_sha256': sha256(request_path), 'candidates': candidates,
        'total_independent_cv_points': 4014405*len(candidates), 'cv_models_bitwise_equal': 5*len(candidates),
        'all_dev_models_bitwise_equal': len(candidates), 'copied_inputs_sha256': inputs_verified,
        'independent_source_files_exact_git_blob_and_sha256': source_files,
        'portable_receipts_sha256': portable, 'seal_rows': 0, 'reduced_new_uses': 0, 'public_submissions': 0,
        'limitations': marker['environment']+' '+top['source_origin_guard_scope'],
        'quality_authority': 'Original five-fold DEV CV only. All-DEV predictions are fingerprints, not validation scores.'}
    write_json(out/'RESULTS.json', result)
    checkpoint(ROOT, 'independent raw reproduction collection', [f'{len(candidates)} original five-fold CV reproductions', f'{len(candidates)} independent complete-DEV model and feature/target fingerprint matches'],
        ['finalist robustness', 'long-horizon null stress', 'remaining prospective candidate queues'],
        artifacts=[(out/'RESULTS.json').relative_to(ROOT).as_posix()])
    print('INDEPENDENT RAW AUDIT PASS', len(candidates), 'candidates', result['cv_models_bitwise_equal'], 'CV models', flush=True)


if __name__ == '__main__':
    main()
