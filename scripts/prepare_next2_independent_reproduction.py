"""Prepare a genuinely fresh raw-data training workspace only in final audit."""
from pathlib import Path
import sys
import json
import shutil
import subprocess
import datetime
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.next2.session import clock, checkpoint
from src.utils.artifacts import write_json, sha256, utc_now


def main():
    timing = clock()
    if not timing['minimum_research_met'] or not timing['audit_started']:
        raise RuntimeError('Independent final training requires six research hours and an active separate audit')
    out = ROOT/'artifacts/next2/independent_reproduction'; out.mkdir(parents=True, exist_ok=True)
    if (out/'PREPARATION.json').exists() or (out/'START.json').exists():
        raise RuntimeError('Existing independent preparation must be reviewed; never silently replace it')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    workspace = ROOT.parent/('SB_NEXT2_REPRO_'+stamp)
    if workspace.exists(): raise RuntimeError('Fresh sibling workspace required')
    write_json(out/'START.json', {'status': 'PREPARING', 'created_utc': utc_now(), 'workspace': str(workspace),
        'script_sha256': sha256(Path(__file__)), 'timing': timing})
    workspace.mkdir(); payload = workspace/'payload'; payload.mkdir()
    files = [p for p in (ROOT/'src').rglob('*.py')]
    files += [ROOT/p for p in ('next2_canonical_distilled_main.py', 'next2_shared_performance_main.py',
        'scripts/check_next2_independent_training.py', 'requirements-lock.txt',
        'configs/next2/DEPLOYMENT_RESEARCH_DISTILLED.json', 'configs/next2/DEPLOYMENT_RESEARCH_PERFORMANCE.json',
        'artifacts/validation/SPLIT_LOCK_REAL.json', 'artifacts/validation/GROUP_ASSIGNMENTS.parquet',
        'SEAL_PARTITION_500x4_LOCK.json', 'REDUCED_TEST_USAGE_LEDGER.json')]
    seal = json.loads((ROOT/'SEAL_PARTITION_500x4_LOCK.json').read_text())
    files += [ROOT/v['ids_file'] for v in seal['partitions'].values()]
    files += [ROOT/'data/raw'/name for name in ('X_train.parquet', 'y_train_index.parquet')]
    manifest = {}
    for path in sorted(set(files)):
        relative = path.relative_to(ROOT); target = payload/relative
        if path.is_symlink(): raise RuntimeError('Ordinary source/data files required')
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
        digest = sha256(path); assert sha256(target) == digest
        manifest[relative.as_posix()] = digest
    expected = {}
    j_training = json.loads((ROOT/'artifacts/next/refits/J102_SERIAL_VARIANCE_512/cache_research_v1/model.json').read_text())['training']
    c_training = json.loads((ROOT/'artifacts/next2/refits/C205_DEPENDENCE/cache_v1/RESULTS.json').read_text())['training']
    for role, variant in (('DISTILLED', 'DISTILLED_tree_used_v1'), ('PERFORMANCE', 'PERFORMANCE_shared_v1')):
        reference = ROOT/'artifacts/next2/canonical_deployment'/variant/'model.sbnext2'
        target = payload/'expected'/role/'model.sbnext2'; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(reference, target); assert sha256(target) == sha256(reference)
        api = 'canonical_api_all_dev' if role == 'DISTILLED' else 'shared_public_api_all_dev'
        receipt = ROOT/'artifacts/next2'/api/role/'RESULTS.json'
        r = json.loads(receipt.read_text()); assert r['status'] == 'PASS'
        expected[role] = {'canonical_model_sha256': sha256(reference), 'model_relative': target.relative_to(payload).as_posix(),
            'all_dev_prediction_sha256': r['output_fingerprint_sha256'], 'series': r['series'], 'points': r['points'],
            'component_training': {'single': j_training} if role == 'DISTILLED' else {'primary': j_training, 'complement': c_training},
            'reference_model_source': reference.relative_to(ROOT).as_posix(), 'reference_api_receipt_sha256': sha256(receipt)}
    write_json(payload/'expected/EXPECTED.json', expected)
    write_json(payload/'COPIED_SOURCE_MANIFEST.json', {'files_sha256': manifest, 'created_utc': utc_now(),
        'source_origin': str(ROOT), 'expected_models_are_separate_and_forbidden_before_training': True})
    environment = workspace/'environment'
    with (out/'ENVIRONMENT_SETUP.log').open('w', encoding='utf-8') as log:
        subprocess.run([sys.executable, '-m', 'venv', str(environment)], check=True, stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        python = environment/'Scripts/python.exe'
        subprocess.run([str(python), '-m', 'pip', 'install', '--disable-pip-version-check', '--require-virtualenv', '--no-input',
            '-r', str(payload/'requirements-lock.txt')], check=True, stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        subprocess.run([str(python), '-m', 'pip', 'check'], check=True, stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    report = {'status': 'PREPARED', 'created_utc': utc_now(), 'workspace': str(workspace), 'payload': str(payload),
        'python': str(python), 'origin': str(ROOT), 'copied_files': len(manifest), 'ordinary_file_copies': True,
        'copied_manifest_sha256': sha256(payload/'COPIED_SOURCE_MANIFEST.json'),
        'requirements_sha256': sha256(payload/'requirements-lock.txt'),
        'setup_log_sha256': sha256(out/'ENVIRONMENT_SETUP.log'), 'training_fits_so_far': 0,
        'script_sha256': sha256(Path(__file__)), 'research_time_gate': timing,
        'protocol_sha256': sha256(ROOT/'docs/NEXT2_INDEPENDENT_TRAINING_PROTOCOL.md')}
    write_json(out/'PREPARATION.json', report)
    checkpoint('Prepared independent raw training workspace in separate final audit', ['Actual public train and complete inference fingerprints for both roles'])
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
