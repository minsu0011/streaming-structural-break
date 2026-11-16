"""Extract the actual verified Desktop archive to a new folder and replay both public roles."""
from pathlib import Path, PurePosixPath
import sys
import json
import zipfile
import hashlib
import datetime
import subprocess
import shutil
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.next2.session import require_handoff_time
from src.utils.artifacts import write_json, sha256, utc_now


def main():
    require_handoff_time()
    base = ROOT/'artifacts/next2/handoff'; receipt_path = base/'DESKTOP_ZIP_VERIFICATION_NEXT2.json'
    receipt = json.loads(receipt_path.read_text()); assert receipt['status'] == 'PASS'
    archive_path = Path(receipt['archive']); assert sha256(archive_path) == receipt['sha256']
    out = base/'actual_archive_replay'
    if out.exists(): raise RuntimeError('Actual archive replay already started; preserve and inspect it')
    out.mkdir()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination = (ROOT.parent/('SB_NEXT2_ZIP_REPLAY_'+stamp)).resolve(); assert not destination.exists()
    assert destination.parent == ROOT.parent.resolve(); destination.mkdir()
    write_json(out/'START.json', {'created_utc': utc_now(), 'archive_sha256': receipt['sha256'], 'destination': str(destination),
        'script_sha256': sha256(Path(__file__))})
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        manifest_bytes = archive.read('SOURCE_MANIFEST_NEXT2.json'); manifest = json.loads(manifest_bytes)
        expected = {name: item['sha256'] for name, item in manifest['files'].items()}
        expected['SOURCE_MANIFEST_NEXT2.json'] = hashlib.sha256(manifest_bytes).hexdigest()
        assert expected['SOURCE_MANIFEST_NEXT2.json'] == receipt['manifest_sha256']
        assert set(archive.namelist()) == set(expected)
        for name in archive.namelist():
            posix = PurePosixPath(name)
            assert not posix.is_absolute() and '..' not in posix.parts and '\\' not in name
            target = destination.joinpath(*posix.parts).resolve(); assert target.is_relative_to(destination)
            content = archive.read(name); assert hashlib.sha256(content).hexdigest() == expected[name]
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream: stream.write(content)
    write_json(destination/'PORTABLE_COPY_MANIFEST.json', {'files_sha256': expected, 'archive_sha256': receipt['sha256'],
        'source_root': str(ROOT), 'raw_competition_inputs_included': False})
    preparation = json.loads((ROOT/'artifacts/next2/independent_reproduction/PREPARATION_READY.json').read_text())
    command = [preparation['python'], '-I', str(destination/'scripts/check_next2_portable_copy.py'), '--origin', str(ROOT)]
    with (out/'RUN.log').open('w', encoding='utf-8') as stream:
        completed = subprocess.run(command, cwd=destination, stdout=stream, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if completed.returncode:
        write_json(out/'FAILURE.json', {'status': 'FAIL', 'exit_code': completed.returncode, 'log_sha256': sha256(out/'RUN.log')})
        raise RuntimeError('Actual archive replay failed; all outputs preserved')
    report = json.loads((destination/'PORTABLE_COPY_RESULT.json').read_text()); assert report['status'] == 'PASS'
    assert sha256(archive_path) == receipt['sha256']
    for name in ('PORTABLE_COPY_RESULT.json', 'PORTABLE_API_REPLAY.json'): shutil.copyfile(destination/name, out/name)
    write_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(), 'archive': str(archive_path),
        'archive_sha256': receipt['sha256'], 'archive_entries_rehashed': len(expected), 'destination': str(destination),
        'public_API_points': report['public_API_points'], 'bootstrap_scopes': report['bootstrap_scopes'],
        'replay_report_sha256': sha256(out/'PORTABLE_COPY_RESULT.json'), 'original_project_access_attempts': [],
        'raw_competition_inputs_included': False, 'new_fits': 0, 'clock': require_handoff_time()})
    receipt.update(actual_archive_extraction_and_replay='PASS', actual_archive_replay_sha256=sha256(out/'RESULTS.json'),
        final_verification_utc=utc_now(), final_clock=require_handoff_time())
    write_json(receipt_path, receipt)
    print('NEXT2 ACTUAL DESKTOP ZIP REPLAY PASS', len(expected), report['public_API_points'], flush=True)


if __name__ == '__main__': main()
