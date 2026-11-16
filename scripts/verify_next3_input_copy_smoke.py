"""Check the actual final input plan without copying observations or fitting."""
from pathlib import Path
import datetime
import argparse
import os
import shutil
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, atomic_json, digest, progress, read_json, utc_now
from scripts.prepare_next3_independent_reproduction import planned_input_files, selected_performance


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--label', default='v1'); args = parser.parse_args()
    if not args.label.isidentifier(): raise ValueError('Unique simple attempt label required')
    out = BASE/'maintenance'/('raw_input_copy_smoke_'+args.label); out.mkdir(parents=True, exist_ok=False)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    workspace = ROOT.parent/('SB_NEXT3_MAINT_INPUT_SMOKE_'+stamp)
    workspace.mkdir(exist_ok=False); payload = workspace/'payload'; payload.mkdir()
    entry, extra = selected_performance(); assert entry == 'next3_lean_performance_main'
    raw = {ROOT/'data/raw/X_train.parquet', ROOT/'data/raw/y_train_index.parquet'}
    plan = planned_input_files(extra); assert raw.issubset(plan)
    files = [path for path in plan if path not in raw]
    files.append(ROOT/'scripts/check_next3_input_copy_smoke.py')
    models = {'DISTILLED': ROOT/'artifacts/next2/canonical_deployment/DISTILLED_tree_used_v1',
              'PERFORMANCE': BASE/'canonical_deployment/PERFORMANCE_lean_shared_v1'}
    for directory in models.values(): files += [directory/'model.json', directory/'model.sbnext2']
    manifest = {}
    for path in sorted(set(files)):
        assert not path.is_symlink() and not path.is_junction()
        relative = path.relative_to(ROOT); assert relative.parts[0] != 'data'
        target = payload/relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target); assert digest(target) == digest(path)
        manifest[relative.as_posix()] = digest(path)
    assert not (payload/'data').exists()
    atomic_json(payload/'SMOKE_INPUT_MANIFEST.json', {'created_utc': utc_now(), 'files_sha256': manifest,
        'entries': {'DISTILLED': 'next2_canonical_distilled_main', 'PERFORMANCE': entry},
        'models': {role: directory.relative_to(ROOT).as_posix() for role, directory in models.items()},
        'excluded_real_raw_files': [path.relative_to(ROOT).as_posix() for path in sorted(raw)],
        'source_plan_sha256': digest(ROOT/'scripts/prepare_next3_independent_reproduction.py'),
        'scope': 'Maintenance-only plan check; no fresh final environment or raw reproduction.'}, immutable=True)
    command = [sys.executable, '-I', '-u', str(payload/'scripts/check_next3_input_copy_smoke.py'), '--origin', str(ROOT)]
    atomic_json(out/'START.json', {'created_utc': utc_now(), 'workspace': str(workspace), 'command': command,
        'new_fits_planned': 0, 'real_raw_inputs_excluded': True, 'copied_files': len(manifest)}, immutable=True)
    env = dict(os.environ, CRUNCH_AUTO_MONKEY_PATCH='false')
    with (out/'RUN.log').open('x', encoding='utf-8') as log:
        result = subprocess.run(command, cwd=payload, env=env, stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if result.returncode:
        atomic_json(out/'FAILURE.json', {'created_utc': utc_now(), 'returncode': result.returncode,
            'all_attempt_files_preserved': True, 'log_sha256': digest(out/'RUN.log')}, immutable=True)
        raise RuntimeError('Copied-input smoke failed; preserve original evidence')
    report = read_json(workspace/'RESULTS.json')
    assert report['status'] == 'PASS_ZERO_FIT_COPIED_INPUT_SMOKE' and report['new_fits'] == 0
    assert set(report['roles']) == {'DISTILLED', 'PERFORMANCE'} and not report['fresh_environment_claimed']
    for relative, expected in manifest.items():
        assert digest(payload/relative) == digest(ROOT/relative) == expected
    shutil.copyfile(payload/'SMOKE_INPUT_MANIFEST.json', out/'INPUT_MANIFEST.json')
    shutil.copyfile(workspace/'RESULTS.json', out/'CHILD_RESULT.json')
    atomic_json(out/'RESULTS.json', {'status': 'PASS_ZERO_FIT_COPIED_INPUT_SMOKE', 'created_utc': utc_now(),
        'workspace': str(workspace), 'roles_checked': 2, 'copied_files': len(manifest),
        'project_modules_loaded_from_copy': len(report['project_modules']), 'new_fits': 0,
        'real_raw_inputs_copied_or_read': False, 'fresh_final_raw_reproduction_completed': False,
        'source_sha256': {path.relative_to(ROOT).as_posix(): digest(path) for path in
            [Path(__file__), ROOT/'scripts/check_next3_input_copy_smoke.py', ROOT/'scripts/prepare_next3_independent_reproduction.py']},
        'files_sha256': {p.name: digest(p) for p in out.iterdir() if p.is_file()},
        'scope': report['scope']}, immutable=True)
    progress('Final raw input-plan smoke passed', 'Two copied public entries reject empty training before fit and reach initial None without dataset access; actual final fresh raw reproduction remains required.')
    print('NEXT3 COPY PLAN SMOKE PASS', len(manifest), flush=True)


if __name__ == '__main__': main()
