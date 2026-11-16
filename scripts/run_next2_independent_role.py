"""Launch an already prepared fresh-environment audit and retain its receipts."""
from pathlib import Path
import sys
import argparse
import json
import shutil
import subprocess
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.next2.session import clock, checkpoint
from src.utils.artifacts import write_json, sha256, utc_now


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--role', choices=['DISTILLED', 'PERFORMANCE'], required=True); args = parser.parse_args()
    timing = clock()
    if not timing['minimum_research_met'] or not timing['audit_started']:
        raise RuntimeError('Independent final training belongs to the separate final audit')
    base = ROOT/'artifacts/next2/independent_reproduction'
    preparation_path = base/'PREPARATION_READY.json'
    preparation = json.loads(preparation_path.read_text()); assert preparation['status'] == 'PREPARED'
    payload = Path(preparation['payload']); workspace = Path(preparation['workspace'])
    assert payload == workspace/'payload' and not workspace.is_relative_to(ROOT)
    assert sha256(payload/'COPIED_SOURCE_MANIFEST.json') == preparation['copied_manifest_sha256']
    out = base/args.role
    if out.exists(): raise RuntimeError('Independent audit role already started; preserve it and review before any revision')
    out.mkdir()
    command = [preparation['python'], '-I', str(payload/'scripts/check_next2_independent_training.py'),
        '--origin', str(ROOT), '--role', args.role]
    write_json(out/'START.json', {'created_utc': utc_now(), 'command': command,
        'preparation_sha256': sha256(preparation_path), 'script_sha256': sha256(Path(__file__))})
    with (out/'RUN.log').open('w', encoding='utf-8') as stream:
        completed = subprocess.run(command, cwd=payload, stdout=stream, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    source = payload/'artifacts/independent'/args.role
    if completed.returncode != 0:
        write_json(out/'FAILURE.json', {'status': 'FAIL', 'exit_code': completed.returncode, 'created_utc': utc_now(),
            'log_sha256': sha256(out/'RUN.log'), 'child_directory': str(source)})
        raise RuntimeError('Independent child failed; its full log and partial artifacts are preserved')
    r = json.loads((source/'RESULTS.json').read_text()); assert r['status'] == 'PASS'
    for path in source.rglob('*'):
        if not path.is_file() or path.suffix not in ('.json', '.sbnext2'): continue
        target = out/path.relative_to(source); target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
        assert sha256(target) == sha256(path)
    write_json(out/'COPY_RECEIPT.json', {'status': 'PASS', 'created_utc': utc_now(), 'source_directory': str(source),
        'files_sha256': {p.relative_to(out).as_posix(): sha256(p) for p in out.rglob('*') if p.is_file()},
        'original_source_and_data_untouched': True, 'new_fits': r['new_fits']})
    checkpoint('Independent fresh-environment canonical training '+args.role,
        ['Finish separate final audit and complete verified GPT handoff'], artifacts=[(out/'RESULTS.json').relative_to(ROOT).as_posix()])
    print('NEXT2 INDEPENDENT ROLE PASS', args.role, r['new_fits'], r['points'], flush=True)


if __name__ == '__main__': main()
