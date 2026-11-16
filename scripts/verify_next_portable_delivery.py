"""Preflight the exact handoff allowlist in copied files and a fresh environment.

This is additional final-audit work. It does not create the Desktop ZIP, fit a
model, score a holdout, or use the original project during child inference.
"""
from pathlib import Path
import json
import os
import re
import shutil
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.package_next_handoff import collect_payload
from scripts.train_next_final_roles import FAMILIES
from src.next.research_clock import ResearchClock
from src.utils.artifacts import write_json, sha256, utc_now


def main():
    clock = ResearchClock(ROOT).snapshot()
    if not clock['minimum_research_met'] or not clock['final_audit_started']:
        raise RuntimeError('Portable delivery preflight belongs to the additional final audit')
    out = ROOT/'artifacts/next/portable_delivery'
    workspace = ROOT/'.next-portable-v3'
    if workspace.exists() or out.exists():
        raise RuntimeError('Preserve the first portable preflight; explicitly review any partial failure')
    out.mkdir(parents=True)
    workspace.mkdir(parents=True)
    tick = time.perf_counter()
    selection = json.loads((ROOT/'DEV_CANDIDATES_NEXT.json').read_text(encoding='utf-8'))
    paths, files, model_files, total_bytes = collect_payload(selection)
    payload, environment = workspace/'payload', workspace/'environment'
    if max(len(str(payload/path.relative_to(ROOT))) for path in paths) >= 260:
        raise RuntimeError('Choose a shorter workspace before copying the package')
    for source in paths:
        relative = source.relative_to(ROOT)
        destination = payload/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as reader, destination.open('xb') as writer:
            shutil.copyfileobj(reader, writer)
        if sha256(destination) != files[relative.as_posix()]['sha256']:
            raise RuntimeError('A copied payload member changed')
    manifest_path = out/'COPIED_PAYLOAD_MANIFEST.json'
    write_json(manifest_path, {'status': 'PASS', 'created_utc': utc_now(), 'files': files,
        'file_count': len(files), 'bytes': total_bytes, 'model_files_sha256': model_files,
        'all_paths_copied_without_symlinks_or_junctions': True,
        'scope': 'The same collect_payload allowlist as the final ZIP, before this preflight, final test/audit receipts and recursive SOURCE_MANIFEST are created.'})
    write_json(out/'IDENTITY.json', {'created_utc': utc_now(), 'clock': clock,
        'workspace': str(workspace), 'selection_sha256': sha256(ROOT/'DEV_CANDIDATES_NEXT.json'),
        'driver_sha256': sha256(Path(__file__)),
        'child_sha256': sha256(ROOT/'scripts/check_next_portable_child.py'),
        'allowlist_sha256': sha256(ROOT/'scripts/package_next_handoff.py'),
        'requirements_sha256': sha256(ROOT/'requirements-lock.txt')})
    clean_env = dict(os.environ)
    clean_env.pop('PYTHONPATH', None)
    clean_env.pop('PYTHONHOME', None)
    clean_env['PYTHONNOUSERSITE'] = '1'
    clean_env['PYTHONDONTWRITEBYTECODE'] = '1'
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        clean_env[name] = '1'
    subprocess.run([sys.executable, '-m', 'venv', str(environment)], cwd=workspace, env=clean_env, check=True)
    python = environment/'Scripts/python.exe'
    command = [str(python), '-m', 'pip', 'install', '--disable-pip-version-check', '--no-input', '-r', str(payload/'requirements-lock.txt')]
    with (out/'PIP_INSTALL.txt').open('w', encoding='utf-8') as stream:
        process = subprocess.Popen(command, cwd=workspace, env=clean_env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            if 'Successfully installed' in line or 'ERROR' in line:
                print(line.rstrip(), flush=True)
        if process.wait() != 0:
            raise RuntimeError('Fresh pinned dependency installation failed; inspect retained PIP_INSTALL.txt')
    installed = json.loads(subprocess.check_output([str(python), '-m', 'pip', 'list', '--format=json', '--disable-pip-version-check'], cwd=workspace, env=clean_env, text=True))
    normalize = lambda value: re.sub(r'[-_.]+', '-', value).lower()
    versions = {normalize(row['name']): row['version'] for row in installed}
    requested = {}
    for line in (payload/'requirements-lock.txt').read_text(encoding='utf-8').splitlines():
        if line.strip() and not line.lstrip().startswith('#'):
            name, version = line.strip().split('==')
            requested[normalize(name)] = version
    if any(versions.get(name) != version for name, version in requested.items()):
        raise RuntimeError('Fresh environment versions differ from the pinned lock')
    pip_check = subprocess.run([str(python), '-m', 'pip', 'check'], cwd=workspace, env=clean_env, text=True, capture_output=True)
    (out/'PIP_CHECK.txt').write_text(pip_check.stdout+pip_check.stderr, encoding='utf-8')
    if pip_check.returncode:
        raise RuntimeError('Fresh environment dependency consistency failed')
    write_json(out/'ENVIRONMENT.json', {'status': 'PASS', 'python': str(python), 'installed': installed,
        'all_pinned_versions_exact': True, 'requested_distribution_count': len(requested), 'pip_check_exit_code': 0})
    reports = []
    for row in selection['candidates']:
        family = row['candidate']['extension']['kind']
        entry_name, _, _, child = FAMILIES[family]
        fixture = json.loads((ROOT/'data/next/final_artifact_contract'/row['role']/'fixture.json').read_text(encoding='utf-8'))
        fixture['model_directory'] = str(payload/row['delivery_model_directory'])
        fixture_path = workspace/'fixtures'/(row['role']+'.json')
        write_json(fixture_path, fixture)
        result_path = workspace/'results'/(row['role']+'.json')
        result_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(python), '-I', str(payload/'scripts/check_next_portable_child.py'),
            '--origin', str(ROOT), '--workspace', str(workspace), '--child', child,
            '--fixture', str(fixture_path), '--output', str(result_path)], cwd=payload, env=clean_env, check=True)
        report = json.loads(result_path.read_text(encoding='utf-8'))
        if report['status'] != 'PASS' or report['entry'] != entry_name or report['model_sha256'] != row['model_sha256'] or report['points'] != 17771:
            raise RuntimeError('Portable selected-role inference coverage differs')
        destination = out/(row['role']+'.json')
        shutil.copyfile(result_path, destination)
        reports.append({'role': row['role'], 'candidate': row['experiment_id'],
            'report': destination.relative_to(ROOT).as_posix(), 'sha256': sha256(destination),
            'fixture_sha256_not_packaged': sha256(fixture_path), 'points': report['points']})
    for relative, record in files.items():
        if sha256(payload/relative) != record['sha256'] or sha256(ROOT/relative) != record['sha256']:
            raise RuntimeError('Source or copied payload changed during preflight: '+relative)
    evidence = [*out.iterdir(), Path(__file__), ROOT/'scripts/check_next_portable_child.py',
        ROOT/'scripts/package_next_handoff.py', ROOT/'requirements-lock.txt']
    write_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(),
        'checks': {'exact_allowlist_copy_sha256': True, 'fresh_pinned_environment_installed': True,
            'dependency_consistency': True, 'all_selected_cold_inference_bitwise': True,
            'origin_project_reads_zero': True, 'project_modules_only_copied_payload': True,
            'readiness_single_pass_no_read_ahead_and_readonly': True, 'copied_payload_unchanged': True},
        'selected_roles': reports, 'points': sum(row['points'] for row in reports),
        'payload_files': len(files), 'elapsed_seconds': time.perf_counter()-tick,
        'evidence_files_sha256': {path.relative_to(ROOT).as_posix(): sha256(path) for path in evidence if path.is_file()},
        'raw_fixtures_in_package': False, 'new_model_fits': 0, 'new_quality_scores': 0,
        'scope': 'New environment and fresh copied package payload on the same OS/hardware. Existing DEV fingerprint fixtures were copied separately outside payload; no seal values. Python audit hooks are not a native OS sandbox.'})
    print('PORTABLE DELIVERY PREFLIGHT PASS', len(reports), flush=True)


if __name__ == '__main__':
    main()
