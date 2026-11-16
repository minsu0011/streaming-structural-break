"""Run a shipped cold-inference contract with origin-project reads denied."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import hashlib
import json
import os
import runpy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--origin', required=True)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--child', required=True)
    parser.add_argument('--fixture', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    origin = Path(args.origin).resolve()
    workspace = Path(args.workspace).resolve()
    payload = Path(__file__).resolve().parents[1]
    if payload != workspace/'payload' or not Path(sys.prefix).resolve().is_relative_to(workspace/'environment'):
        raise RuntimeError('The copied payload and newly created environment are required')
    denied = []
    read_opens = [0]
    def audit(event, arguments):
        if event != 'open':
            return
        raw, mode, flags = arguments
        if not isinstance(raw, (str, bytes, os.PathLike)):
            return
        writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (isinstance(flags, int) and bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
        if writing:
            return  # The unchanged inner cold contract guards all mutations.
        path = Path(os.fsdecode(raw)).resolve()
        if path.is_relative_to(origin) and not path.is_relative_to(workspace):
            denied.append(str(path))
            raise RuntimeError('PORTABLE_ORIGIN_PROJECT_READ_DENIED')
        read_opens[0] += 1
    sys.addaudithook(audit)
    sys.path = [str(payload), *[item for item in sys.path if item and not Path(item).resolve().is_relative_to(origin)]]
    # Isolated-mode sys.path normally includes the fresh environment, which is
    # nested below origin. Restore exactly that workspace's site-packages.
    site_packages = Path(sys.prefix)/'Lib/site-packages'
    sys.path.append(str(site_packages))
    child = payload/args.child
    sys.argv = [str(child), '--fixture', args.fixture, '--output', args.output]
    runpy.run_path(str(child), run_name='__main__')
    report_path = Path(args.output)
    report = json.loads(report_path.read_text(encoding='utf-8'))
    modules = {}
    for name, module in list(sys.modules.items()):
        if name != 'src' and not name.startswith('src.') and not name.startswith('next_'):
            continue
        source = getattr(module, '__file__', None)
        if source is None:
            continue
        path = Path(source).resolve()
        if not path.is_relative_to(payload):
            raise RuntimeError('A project module escaped the copied payload: '+name)
        modules[name] = {'relative': path.relative_to(payload).as_posix(),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    if report['status'] != 'PASS' or denied or not modules or report['audit']['filesystem_mutation_attempts']:
        raise RuntimeError('Portable inference contract failed')
    report.update(origin_project_reads_denied_by_hook=True, origin_project_read_attempts=denied,
        project_modules_all_from_payload=True, project_modules=modules, guarded_read_opens=read_opens[0],
        fresh_environment=sys.prefix, payload_directory=str(payload),
        scope='Fresh copied source and selected delivery weights, newly installed pinned dependencies, unchanged cold API contract. Python audit-hook boundary, not an OS sandbox.')
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print('PORTABLE COLD CONTRACT PASS', report['entry'], report['points'], flush=True)


if __name__ == '__main__':
    main()
