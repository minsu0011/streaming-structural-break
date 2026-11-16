"""Run the fresh public-training reproductions and retain compact evidence only."""
from pathlib import Path
import sys
import os
import shutil
import subprocess
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, read_json, atomic_json, digest, utc_now, progress
from scripts.prepare_next3_independent_reproduction import time_gate


def main():
    timing = time_gate(); out = BASE/'independent_reproduction'; preparation = read_json(out/'PREPARATION.json')
    assert preparation['status'] == 'PREPARED'
    payload = Path(preparation['payload']); workspace = Path(preparation['workspace']); python = Path(preparation['python'])
    assert not workspace.resolve().is_relative_to(ROOT.resolve()) and payload.parent == workspace
    manifest_path = payload/'COPIED_SOURCE_MANIFEST.json'
    assert digest(manifest_path) == preparation['copied_manifest_sha256']
    manifest = read_json(manifest_path)['files_sha256']
    for name, sha in manifest.items():
        assert digest(payload/name) == sha
        if name != 'REPRODUCTION_ENTRY_LOCK.json': assert digest(ROOT/name) == sha, name
    summaries = []
    for role in ('DISTILLED', 'PERFORMANCE'):
        target = out/role
        if target.exists(): raise RuntimeError('Existing role reproduction evidence requires explicit review, not overwrite')
        target.mkdir()
        command = [str(python), '-I', '-u', str(payload/'scripts/check_next3_independent_training.py'), '--origin', str(ROOT), '--role', role]
        child_environment = dict(os.environ, CRUNCH_AUTO_MONKEY_PATCH='false')
        atomic_json(target/'START.json', {'created_utc': utc_now(), 'role': role, 'command': command,
            'preparation_sha256': digest(out/'PREPARATION.json'), 'time_gate': timing,
            'isolated_python_mode': True, 'SDK_auto_monkey_patch_disabled_for_child': True}, immutable=True)
        progress('Independent raw public training', role+' using copied source and fresh pinned environment', phase='FINAL_AUDIT')
        with (target/'RUN.log').open('x', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=payload, env=child_environment, stdout=log, stderr=subprocess.STDOUT,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if result.returncode:
            atomic_json(target/'FAILURE.json', {'status': 'FAIL', 'created_utc': utc_now(), 'returncode': result.returncode,
                'log_sha256': digest(target/'RUN.log'), 'all_attempt_files_preserved': True}, immutable=True)
            raise RuntimeError('Independent raw reproduction failed for '+role)
        source = payload/'artifacts/independent'/role; report = read_json(source/'RESULTS.json')
        assert report['status'] == 'PASS' and report['role'] == role and report['series'] == 8000 and report['points'] == 4014405
        assert report['new_fits'] == (1 if role == 'DISTILLED' else 2)
        assert report['seal_values_accessed'] == report['seal_rows_fitted'] == 0
        for name in ('RESULTS.json', 'model/model.sbnext2', 'model/model.json'):
            destination = target/name; destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source/name, destination); assert digest(destination) == digest(source/name)
        summaries.append({'role': role, 'entry_module': report['entry_module'], 'runtime_variant': report['runtime_variant'],
            'canonical_model_sha256': report['canonical_model_sha256'], 'all_dev_prediction_sha256': report['all_dev_prediction_sha256'],
            'series': report['series'], 'points': report['points'], 'raw_reproduction_fits': report['new_fits'],
            'training_seconds': report['training_seconds'], 'total_seconds': report['total_seconds'],
            'result_sha256': digest(target/'RESULTS.json'), 'log_sha256': digest(target/'RUN.log')})
    atomic_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(), 'roles': summaries,
        'workspace': str(workspace), 'fresh_environment': True, 'raw_copies_only_no_prior_feature_cache': True,
        'scientific_recipe_count': 2, 'raw_reproduction_fits': 3, 'new_candidate_search_fits': 0,
        'all_seals_unopened': True, 'reduced_new_usage': 0, 'public_submission_new': 0,
        'preparation_sha256': digest(out/'PREPARATION.json'), 'script_sha256': digest(Path(__file__)),
        'boundary': 'Python audit hooks plus module/source/byte verification; not an OS sandbox'}, immutable=True)
    progress('Both raw public reproductions complete', 'Canonical model bytes and all 8,028,810 DEV predictions reproduced', phase='FINAL_AUDIT')
    print('NEXT3 INDEPENDENT RAW REPRODUCTION PASS BOTH ROLES', flush=True)


if __name__ == '__main__': main()
