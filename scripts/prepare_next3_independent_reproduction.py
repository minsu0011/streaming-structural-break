"""Prepare ordinary raw copies and a fresh environment only in NEXT3 final audit."""
from pathlib import Path
import sys
import time
import datetime
import shutil
import subprocess
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, read_json, atomic_json, digest, utc_now, progress
from src.next3.offline_environment import copy_verified_wheels


def time_gate():
    session = read_json(BASE/'SESSION.json')
    wall = time.time()-datetime.datetime.fromisoformat(session['started_utc']).timestamp()
    if wall < 40200 or not session.get('audit_started_utc'):
        raise RuntimeError('NEXT3 raw reproduction requires 11h10 research and a separately active final audit')
    started = datetime.datetime.fromisoformat(session['audit_started_utc']).timestamp()
    if started < datetime.datetime.fromisoformat(session['started_utc']).timestamp()+40200:
        raise RuntimeError('NEXT3 final audit timestamp predates the research gate')
    if started > time.time():
        raise RuntimeError('NEXT3 final audit timestamp is in the future')
    return {'wall_seconds': wall, 'research_minimum_seconds': 40200, 'audit_started_utc': session['audit_started_utc']}


def selected_performance():
    lean_path = ROOT/'configs/next3/LEAN_RUNTIME_SELECTION.json'
    if lean_path.exists():
        lean = read_json(lean_path)
        if lean.get('status') in ('PROVISIONAL_ENGINEERING_ELIGIBLE', 'FINAL_ACCEPTED'):
            if lean.get('variant') != 'LEAN_PACKED_SHARED': raise RuntimeError('Unknown lean performance variant')
            api = read_json(BASE/'lean_public_api_all_dev/RESULTS.json')
            cold = read_json(BASE/'engineering/lean_shared/cold/RESULTS.json')
            direct = read_json(BASE/'engineering/lean_shared/direct/RESULTS.json')
            if (api['status'] != 'PASS' or api['identity']['variant'] != lean['variant'] or
                    cold['status'] != 'PASS' or not cold['cold_cost_gate_pass'] or
                    direct['status'] != 'PASS' or not direct['direct_cost_gate_pass']):
                raise RuntimeError('Lean public API and both actual cost gates required')
            return 'next3_lean_performance_main', [lean_path, ROOT/'next3_lean_performance_main.py',
                *[ROOT/'configs/next3'/name for name in ('LEAN_FEATURE_CONTRACT.json', 'LEAN_NUMERIC_BUILD.json', 'LEAN_RUNTIME_INPUTS.json')]]
        if lean.get('status') not in ('VALIDATION_ONLY', 'REJECTED_COST', 'REJECTED_VALIDATION'):
            raise RuntimeError('Unrecognized lean selection state')
    selection_path = ROOT/'configs/next3/PACKED_RUNTIME_SELECTION.json'
    if selection_path.exists():
        selection = read_json(selection_path)
        if selection['status'] not in ('PROVISIONAL_ENGINEERING_ELIGIBLE', 'FINAL_ACCEPTED'):
            raise RuntimeError('Ineligible packed runtime cannot enter final reproduction')
        api = read_json(BASE/'packed_public_api_all_dev/RESULTS.json')
        if api['status'] != 'PASS': raise RuntimeError('Actual all-DEV packed public API proof required')
        if api['identity']['variant'] != selection['variant']:
            raise RuntimeError('Public API evidence and selected packed variant differ')
        return 'next3_packed_performance_main', [selection_path, ROOT/'next3_packed_performance_main.py']
    return 'next2_shared_performance_main', []


def planned_input_files(selected_extra):
    """Return the unchanged source/config/guard/raw file plan without copying it."""
    files = list((ROOT/'src').rglob('*.py'))
    files += [ROOT/name for name in ('next2_canonical_distilled_main.py', 'next2_shared_performance_main.py',
        'scripts/check_next3_independent_training.py', 'requirements-lock.txt',
        'configs/next2/DEPLOYMENT_RESEARCH_DISTILLED.json', 'configs/next2/DEPLOYMENT_RESEARCH_PERFORMANCE.json',
        'configs/features.json', 'configs/arch_input.json', 'configs/ar_residual_input.json',
        'configs/historical_context.json', 'configs/historical_quantiles.json',
        'configs/rank_gaussian.json', 'configs/magnitude_dependence.json', 'ROBUSTNESS_SPLIT_LOCK.json',
        'artifacts/validation/SPLIT_LOCK_REAL.json', 'artifacts/validation/GROUP_ASSIGNMENTS.parquet',
        'SEAL_PARTITION_500x4_LOCK.json', 'REDUCED_TEST_USAGE_LEDGER.json')]
    seal = read_json(ROOT/'SEAL_PARTITION_500x4_LOCK.json')
    files += [ROOT/part['ids_file'] for part in seal['partitions'].values()]
    files += [ROOT/'data/raw'/name for name in ('X_train.parquet', 'y_train_index.parquet')]
    return sorted(set(files + selected_extra))


def main():
    timing = time_gate(); out = BASE/'independent_reproduction'; out.mkdir(parents=True, exist_ok=True)
    if (out/'START.json').exists() or (out/'PREPARATION.json').exists():
        raise RuntimeError('Preserve an earlier independent reproduction attempt')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    workspace = ROOT.parent/('SB_NEXT3_REPRO_'+stamp)
    if workspace.exists(): raise RuntimeError('A new ordinary sibling workspace is required')
    performance_entry, selected_extra = selected_performance()
    entries = {'DISTILLED': 'next2_canonical_distilled_main', 'PERFORMANCE': performance_entry}
    atomic_json(out/'START.json', {'status': 'PREPARING', 'created_utc': utc_now(), 'workspace': str(workspace),
        'script_sha256': digest(Path(__file__)), 'timing': timing, 'entries': entries}, immutable=True)
    workspace.mkdir(); payload = workspace/'payload'; payload.mkdir()
    manifest = {}
    for path in planned_input_files(selected_extra):
        relative = path.relative_to(ROOT); target = payload/relative
        if path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction()):
            raise RuntimeError('Ordinary input files required')
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
        assert digest(target) == digest(path); manifest[relative.as_posix()] = digest(path)
    atomic_json(payload/'REPRODUCTION_ENTRY_LOCK.json', {'entries': entries, 'created_utc': utc_now(),
        'scientific_references': ['J102_SERIAL_VARIANCE_512', 'BL2_C205_DEPENDENCE_80'],
        'no_new_scientific_selection': True}, immutable=True)
    manifest['REPRODUCTION_ENTRY_LOCK.json'] = digest(payload/'REPRODUCTION_ENTRY_LOCK.json')
    j_training = read_json(ROOT/'artifacts/next/refits/J102_SERIAL_VARIANCE_512/cache_research_v1/model.json')['training']
    c_training = read_json(ROOT/'artifacts/next2/refits/C205_DEPENDENCE/cache_v1/RESULTS.json')['training']
    expected = {}
    for role, variant in (('DISTILLED', 'DISTILLED_tree_used_v1'), ('PERFORMANCE', 'PERFORMANCE_shared_v1')):
        reference = ROOT/'artifacts/next2/canonical_deployment'/variant/'model.sbnext2'
        target = payload/'expected'/role/'model.sbnext2'; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(reference, target); assert digest(target) == digest(reference)
        api = 'canonical_api_all_dev' if role == 'DISTILLED' else 'shared_public_api_all_dev'
        receipt = ROOT/'artifacts/next2'/api/role/'RESULTS.json'; record = read_json(receipt); assert record['status'] == 'PASS'
        expected[role] = {'canonical_model_sha256': digest(reference), 'model_relative': target.relative_to(payload).as_posix(),
            'all_dev_prediction_sha256': record['output_fingerprint_sha256'], 'series': record['series'], 'points': record['points'],
            'component_training': {'single': j_training} if role == 'DISTILLED' else {'primary': j_training, 'complement': c_training},
            'reference_model_source': reference.relative_to(ROOT).as_posix(), 'reference_api_receipt_sha256': digest(receipt)}
    atomic_json(payload/'expected/EXPECTED.json', expected, immutable=True)
    atomic_json(payload/'COPIED_SOURCE_MANIFEST.json', {'files_sha256': manifest, 'created_utc': utc_now(),
        'source_origin': str(ROOT), 'expected_models_are_separate_and_forbidden_before_training': True}, immutable=True)
    wheel_copy = copy_verified_wheels(ROOT, BASE, workspace/'packages')
    if wheel_copy is not None:
        atomic_json(out/'OFFLINE_WHEEL_COPIES.json', wheel_copy, immutable=True)
    environment = workspace/'environment'
    with (out/'ENVIRONMENT_SETUP.log').open('x', encoding='utf-8') as log:
        flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        subprocess.run([sys.executable, '-m', 'venv', str(environment)], check=True, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
        python = environment/'Scripts/python.exe'
        install = [str(python), '-m', 'pip', '--isolated', '--disable-pip-version-check', 'install',
            '--require-virtualenv', '--no-input', '--no-cache-dir']
        if wheel_copy is not None:
            install += ['--no-index', '--find-links', wheel_copy['directory']]
        else:
            install += ['--index-url', 'https://pypi.org/simple']
        install += ['-r', str(payload/'requirements-lock.txt')]
        subprocess.run(install, check=True, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
        subprocess.run([str(python), '-m', 'pip', 'check'], check=True, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
    atomic_json(out/'PREPARATION.json', {'status': 'PREPARED', 'created_utc': utc_now(), 'workspace': str(workspace),
        'payload': str(payload), 'python': str(python), 'origin': str(ROOT), 'entries': entries, 'copied_files': len(manifest),
        'ordinary_file_copies': True, 'copied_manifest_sha256': digest(payload/'COPIED_SOURCE_MANIFEST.json'),
        'requirements_sha256': digest(payload/'requirements-lock.txt'), 'setup_log_sha256': digest(out/'ENVIRONMENT_SETUP.log'),
        'offline_install_from_verified_ordinary_wheel_copies': wheel_copy is not None,
        'offline_wheel_copy_receipt_sha256': digest(out/'OFFLINE_WHEEL_COPIES.json') if wheel_copy is not None else None,
        'training_fits_so_far': 0, 'script_sha256': digest(Path(__file__)), 'research_time_gate': timing,
        'protocol_sha256': digest(ROOT/'docs/NEXT3_INDEPENDENT_TRAINING_PROTOCOL.md')}, immutable=True)
    progress('NEXT3 final-audit raw workspace prepared', 'Actual public train and complete inference remain required', phase='FINAL_AUDIT')
    print('NEXT3 INDEPENDENT WORKSPACE PREPARED', workspace, flush=True)


if __name__ == '__main__': main()
