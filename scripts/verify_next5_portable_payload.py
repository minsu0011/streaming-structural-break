"""Fresh-process raw-free inference from a separate copied NEXT5 payload."""
from pathlib import Path
import argparse
import os
import re
import shutil
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next5.artifacts import BASE, atomic_json, read_json, digest, utc_now, progress
from src.next5.handoff_payload import collect


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    assert re.fullmatch('[A-Za-z0-9_]+', args.tag)
    session = read_json(BASE / 'SESSION.json')
    assert session.get('audit_started_utc'), 'Portable final replay requires the separate audit phase'
    paths, files, excluded, missing = collect(require_final=False)
    out = BASE / 'portable_verification' / args.tag
    out.mkdir(parents=True, exist_ok=False)
    payload = out / 'payload'
    payload.mkdir()
    for number, source in enumerate(paths):
        relative = source.relative_to(ROOT)
        destination = (payload / relative).resolve()
        assert destination.is_relative_to(payload.resolve())
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        assert digest(destination) == files[relative.as_posix()]['sha256']
        if number % 500 == 0:
            progress('Copying raw-free source and evidence', str(number) + '/' + str(len(paths)), phase='FINAL_AUDIT')
    assert not (payload / 'data').exists()
    atomic_json(out / 'COPY_MANIFEST.json', {'created_utc': utc_now(), 'files': files,
        'missing_final_reports_at_copy': missing, 'excluded': excluded}, immutable=True)
    specifications = (
        ('BL80', 'scripts/verify_next4_cold_api.py', ['--worker', 'BL80', '--output']),
        ('B2', 'scripts/verify_next4_b2_readonly.py', ['--worker-output']),
        ('D2', 'scripts/verify_next4_candidate_api.py', ['--model', 'D2_DISCRETE_SCALE_MIXTURE', '--cold-worker-output']),
    )
    records = []
    for role, script, options in specifications:
        target = out / (role + '_COLD.json')
        environment = dict(os.environ, PYTHONPATH=str(payload), PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        process = subprocess.run([sys.executable, '-B', str(payload / script), *options, str(target)],
            cwd=payload, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=300, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        (out / (role + '.log')).write_text(process.stdout, encoding='utf-8')
        process.check_returncode()
        result = read_json(target)
        original_path = BASE / 'preflight' / (role + '_COLD.json')
        original = read_json(original_path)
        assert result['status'] == original['status'] == 'PASS'
        assert result['prediction_sha256'] == original['prediction_sha256']
        write_field = 'persistent_write_attempts' if role == 'BL80' else 'cold_import_write_attempts'
        assert result[write_field] == result['new_fits'] == result['real_or_seal_rows'] == 0
        records.append({'role': role, 'entry': script, 'prediction_sha256': result['prediction_sha256'],
            'copied_source_file_sha256': digest(payload / script), 'fresh_worker_result_sha256': digest(target),
            'reference_result_sha256': digest(original_path), 'new_fits': 0,
            'separate_source_and_model_directory': True, 'fresh_Python_process': True})
        progress('Portable frozen inference verified', role + '; original synthetic fingerprint and zero writes', phase='FINAL_AUDIT')
        print('NEXT5 RAW-FREE PORTABLE', role, 'PASS', flush=True)
    atomic_json(out / 'RESULTS.json', {'created_utc': utc_now(), 'status': 'PASS', 'records': records,
        'copied_files': len(paths), 'copy_manifest_sha256': digest(out / 'COPY_MANIFEST.json'),
        'fresh_processes': len(records), 'all_copied_file_sha256_verified': True,
        'raw_competition_data_present': False, 'new_fits': 0, 'real_or_seal_rows': 0,
        'original_pinned_interpreter_environment_reused': True, 'new_environment_installation': False,
        'fresh_raw_training_reproduction': False, 'new_NEXT5_deployment_role': False,
        'scope': 'Moved source/model inference in fresh processes, using fixed synthetic inputs. BL80 and D2 retained roles; B2 research reference. Not new raw-to-training reproduction.',
        'source_sha256': digest(Path(__file__)),
        'files_sha256': {p.name: digest(p) for p in out.iterdir() if p.is_file()}}, immutable=True)


if __name__ == '__main__':
    main()
