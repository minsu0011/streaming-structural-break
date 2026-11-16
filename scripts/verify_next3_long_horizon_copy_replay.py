"""Run the optional long replay from ordinary source copies without .npy files."""
from pathlib import Path
import datetime
import os
import shutil
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, atomic_json, digest, progress, read_json, utc_now
from scripts.prepare_next3_independent_reproduction import planned_input_files, selected_performance


def main():
    out = BASE/'maintenance/long_horizon_copy_replay_v1'; out.mkdir(parents=True, exist_ok=False)
    original_result = BASE/'maintenance/lean_long_horizon_parity_v1/RESULTS.json'
    assert read_json(original_result)['status'] == 'PASS'
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    workspace = ROOT.parent/('SB_NEXT3_MAINT_LONG_REPLAY_'+stamp); workspace.mkdir(exist_ok=False)
    payload = workspace/'payload'; payload.mkdir()
    entry, extra = selected_performance(); assert entry == 'next3_lean_performance_main'
    files = [path for path in planned_input_files(extra) if path.relative_to(ROOT).parts[0] != 'data']
    model = BASE/'canonical_deployment/PERFORMANCE_lean_shared_v1'
    files += [ROOT/'scripts/replay_next3_long_horizon.py', ROOT/'scripts/verify_next3_lean_long_horizon_parity.py',
        ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json', BASE/'maintenance/LEAN_LONG_HORIZON_PARITY_PROTOCOL.json',
        model/'model.sbnext2', model/'model.json', original_result,
        original_result.parent/'START.json', original_result.parent/'PER_SYNTHETIC_SERIES.csv']
    manifest = {}
    for path in sorted(set(files)):
        assert path.is_file() and not path.is_symlink() and not path.is_junction()
        relative = path.relative_to(ROOT); assert relative.parts[0] != 'data' and path.suffix != '.npy'
        target = payload/relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target); assert digest(target) == digest(path)
        manifest[relative.as_posix()] = {'sha256':digest(path), 'bytes':path.stat().st_size}
    assert not (payload/'data').exists() and not list(payload.rglob('*.npy'))
    atomic_json(payload/'SOURCE_MANIFEST_NEXT3.json', {'status':'MAINTENANCE_COPY_NOT_FINAL_ARCHIVE',
        'created_utc':utc_now(), 'files':manifest, 'original_long_parity_result_sha256':digest(original_result),
        'fresh_environment':False, 'actual_Desktop_archive':False,
        'scope':'Only source/config/model/guard metadata and compact golden hashes; no observation or saved prediction arrays. Uses the existing pinned environment.'}, immutable=True)
    command = [sys.executable, '-I', '-u', str(payload/'scripts/replay_next3_long_horizon.py'),
        '--model-dir', model.relative_to(ROOT).as_posix(), '--output', 'OPTIONAL_LONG_HORIZON_RESULT.json']
    atomic_json(out/'START.json', {'created_utc':utc_now(), 'workspace':str(workspace), 'command':command,
        'copied_files':len(manifest), 'no_competition_data_or_saved_prediction_arrays':True,
        'source_manifest_sha256':digest(payload/'SOURCE_MANIFEST_NEXT3.json'), 'new_fits':0}, immutable=True)
    progress('Optional raw-free long replay from a source copy','Original source/model and compact per-series hashes only; no .npy files or raw data copied; existing environment explicitly retained')
    with (out/'RUN.log').open('x', encoding='utf-8') as log:
        child = subprocess.run(command, cwd=payload, env=dict(os.environ, CRUNCH_AUTO_MONKEY_PATCH='false'),
            stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if child.returncode:
        atomic_json(out/'FAILURE.json', {'created_utc':utc_now(), 'returncode':child.returncode,
            'log_sha256':digest(out/'RUN.log'), 'all_attempt_files_preserved':True}, immutable=True)
        raise RuntimeError('Optional copied long replay failed; original evidence preserved')
    source = payload/'OPTIONAL_LONG_HORIZON_RESULT.json'; result = read_json(source)
    assert result['status'] == 'PASS' and result['public_API_prediction_points'] == 34603008 and result['synthetic_streams'] == 3072
    assert result['new_fits'] == result['new_quality_scores'] == 0
    assert result['saved_prediction_arrays_needed'] is result['raw_competition_inputs_needed'] is False
    for name, record in manifest.items(): assert digest(payload/name) == digest(ROOT/name) == record['sha256']
    shutil.copyfile(source, out/'CHILD_RESULT.json')
    shutil.copyfile(payload/'SOURCE_MANIFEST_NEXT3.json', out/'INPUT_MANIFEST.json')
    atomic_json(out/'RESULTS.json', {'status':'PASS', 'created_utc':utc_now(), 'workspace':str(workspace),
        'public_API_prediction_points':result['public_API_prediction_points'], 'synthetic_streams':result['synthetic_streams'],
        'source_and_inputs_unchanged':True, 'copied_files':len(manifest),
        'project_modules_loaded_from_copy':len(result['project_modules']), 'new_fits':0, 'new_quality_scores':0,
        'raw_observations_and_saved_prediction_arrays_absent':True, 'fresh_environment_claimed':False,
        'actual_final_archive_replay_claimed':False,
        'source_sha256':{path.relative_to(ROOT).as_posix():digest(path) for path in
            [Path(__file__), ROOT/'scripts/replay_next3_long_horizon.py', ROOT/'src/next3/long_horizon_contract.py']},
        'files_sha256':{path.name:digest(path) for path in out.iterdir() if path.is_file()},
        'scope':'The optional long-horizon command actually ran from an ordinary source copy containing no .npy prediction files and no competition raw data. Existing pinned environment; not final fresh raw training or actual final ZIP verification.'}, immutable=True)
    progress('Optional complete long-horizon source-copy replay passed','34,603,008 input/output hash matches using compact reference CSV only; no saved prediction arrays or new fits')
    print('NEXT3 OPTIONAL LONG-HORIZON SOURCE-COPY REPLAY PASS',result['public_API_prediction_points'],flush=True)


if __name__ == '__main__': main()
