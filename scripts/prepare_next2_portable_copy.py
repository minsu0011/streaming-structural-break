"""Copy the reviewable important-results payload and test it without original/raw inputs."""
from pathlib import Path
import sys
import json
import shutil
import datetime
import subprocess
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
from package_next2_handoff import collect_payload
from src.next2.session import clock
from src.utils.artifacts import write_json, sha256, utc_now


def main():
    assert clock()['audit_started']
    out = ROOT/'artifacts/next2/handoff/portable_copy'; out.mkdir(parents=True, exist_ok=True)
    if (out/'START.json').exists(): raise RuntimeError('Never replace an earlier portable copy')
    paths, files, excluded = collect_payload(require_audit=False)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination = ROOT.parent/('SB_NEXT2_PACKAGE_REVIEW_'+stamp); assert not destination.exists()
    preparation = json.loads((ROOT/'artifacts/next2/independent_reproduction/PREPARATION_READY.json').read_text())
    write_json(out/'START.json', {'created_utc': utc_now(), 'destination': str(destination), 'source_files': len(paths),
        'script_sha256': sha256(Path(__file__)), 'environment': preparation['python']})
    destination.mkdir()
    for path in paths:
        target = destination/path.relative_to(ROOT); target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target); assert sha256(target) == files[path.relative_to(ROOT).as_posix()]['sha256']
    write_json(destination/'PORTABLE_COPY_MANIFEST.json', {'files_sha256': {p: r['sha256'] for p, r in files.items()},
        'source_root': str(ROOT), 'raw_competition_inputs_included': False, 'excluded_raw_records': excluded})
    command = [preparation['python'], '-I', str(destination/'scripts/check_next2_portable_copy.py'), '--origin', str(ROOT)]
    with (out/'RUN.log').open('w', encoding='utf-8') as stream:
        completed = subprocess.run(command, cwd=destination, stdout=stream, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if completed.returncode:
        write_json(out/'FAILURE.json', {'status': 'FAIL', 'exit_code': completed.returncode, 'log_sha256': sha256(out/'RUN.log'), 'destination': str(destination)})
        raise RuntimeError('Portable child failed; copy and logs preserved')
    result = json.loads((destination/'PORTABLE_COPY_RESULT.json').read_text()); assert result['status'] == 'PASS'
    for source, name in [(destination/'PORTABLE_COPY_RESULT.json', 'RESULTS.json'),
        (destination/'PORTABLE_API_REPLAY.json', 'API_REPLAY.json'),
        (destination/'artifacts/next2/report_bootstrap_replay/RESULTS.json', 'BOOTSTRAP_REPLAY.json'),
        (destination/'PORTABLE_COPY_MANIFEST.json', 'COPIED_FILES.json')]: shutil.copyfile(source, out/name)
    print('NEXT2 PORTABLE PACKAGE COPY VERIFIED', str(destination), len(paths), flush=True)


if __name__ == '__main__': main()
