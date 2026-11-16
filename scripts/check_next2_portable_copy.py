"""Child verifies raw-free package replay while original-project reads are denied."""
import os
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import json
import hashlib
import runpy


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--origin', required=True); args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]; origin = Path(args.origin).resolve()
    assert not root.is_relative_to(origin) and not (root/'data').exists()
    sys.path = [str(root), *[p for p in sys.path if p and not Path(p).resolve().is_relative_to(origin)]]
    denied = []; raw_reads = []; network = []
    def audit(event, arguments):
        if event in ('socket.connect', 'socket.connect_ex'): network.append(event); raise RuntimeError('No network in portable replay')
        if event == 'open' and isinstance(arguments[0], (str, bytes, os.PathLike)):
            raw = os.fsdecode(arguments[0])
            if os.path.normcase(os.path.normpath(raw)) == os.path.normcase(os.devnull): return
            path = Path(raw).resolve()
            if path.is_relative_to(origin): denied.append(str(path)); raise RuntimeError('Original project read denied')
            if path.name in ('X_train.parquet', 'y_train_index.parquet'): raw_reads.append(str(path)); raise RuntimeError('No raw competition input permitted')
    sys.addaudithook(audit)
    def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    manifest = json.loads((root/'PORTABLE_COPY_MANIFEST.json').read_text())
    for relative, expected in manifest['files_sha256'].items(): assert digest(root/relative) == expected, relative
    sys.argv = ['replay_next2_portable.py', '--output', str(root/'PORTABLE_API_REPLAY.json')]
    runpy.run_path(str(root/'scripts/replay_next2_portable.py'), run_name='__main__')
    sys.argv = ['replay_next2_report_bootstrap.py']
    runpy.run_path(str(root/'scripts/replay_next2_report_bootstrap.py'), run_name='__main__')
    allowed_change = 'artifacts/next2/report_bootstrap_replay/RESULTS.json'
    checked = 0
    for relative, expected in manifest['files_sha256'].items():
        if relative == allowed_change: continue
        assert digest(root/relative) == expected, relative; checked += 1
    api = json.loads((root/'PORTABLE_API_REPLAY.json').read_text()); assert api['status'] == 'PASS' and api['total_points'] == 524288
    bootstrap = json.loads((root/allowed_change).read_text()); assert bootstrap['status'] == 'PASS' and bootstrap['scope_count'] == 24
    modules = {}
    for name, module in list(sys.modules.items()):
        if name != 'src' and not name.startswith(('src.', 'next2_')): continue
        source = getattr(module, '__file__', None)
        if source:
            path = Path(source).resolve(); assert path.is_relative_to(root), name
            modules[name] = {'file': path.relative_to(root).as_posix(), 'sha256': digest(path)}
    assert not denied and not raw_reads and not network
    result = {'status': 'PASS', 'root': str(root), 'unchanged_payload_files_checked': checked,
        'public_API_points': 524288, 'bootstrap_scopes': 24, 'original_project_access_attempts': denied,
        'raw_competition_file_access_attempts': raw_reads, 'network_attempts': network,
        'project_modules': modules, 'API_replay_sha256': digest(root/'PORTABLE_API_REPLAY.json'),
        'bootstrap_replay_sha256': digest(root/allowed_change), 'copy_manifest_sha256': digest(root/'PORTABLE_COPY_MANIFEST.json'),
        'script_sha256': digest(Path(__file__)), 'new_fits': 0,
        'scope': 'Independent raw-free ordinary-file package copy, pinned environment, actual public inference and aggregate bootstrap replay. Audit hooks are not an OS sandbox.'}
    (root/'PORTABLE_COPY_RESULT.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('NEXT2 RAW-FREE COPIED PACKAGE PASS', checked, flush=True)


if __name__ == '__main__': main()
