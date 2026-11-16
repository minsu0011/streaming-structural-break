"""One fresh process: import and first warmed None, with no dataset access."""
from pathlib import Path
import argparse
import importlib
import json
import os
import sys
import time
import psutil
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'): os.environ[key] = '1'


class NoDatasets:
    def __iter__(self): raise AssertionError('Dataset iterator accessed before first None')
    def __len__(self): raise AssertionError('Dataset length accessed before first None')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--variant', choices=('PACKED_SHARED', 'LEAN_PACKED_SHARED'), required=True)
    parser.add_argument('--output', required=True); args = parser.parse_args()
    output = Path(args.output)
    if output.exists(): raise RuntimeError('Preserve every cold measurement')
    entry = 'next3_packed_performance_main' if args.variant == 'PACKED_SHARED' else 'next3_lean_performance_main'
    folder = 'PERFORMANCE_packed_shared_v1' if args.variant == 'PACKED_SHARED' else 'PERFORMANCE_lean_shared_v1'
    active = [True]; attempts = []; null_opens = []
    def audit(event, arguments):
        if not active[0]: return
        if event.startswith('socket.connect'):
            attempts.append(event); raise RuntimeError('Network forbidden during inference')
        writing = event in ('os.mkdir', 'os.remove', 'os.rmdir', 'os.rename', 'os.chmod', 'os.truncate', 'os.link', 'os.symlink')
        if event == 'open':
            path, mode, flags = arguments
            writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if writing and isinstance(path, (str, bytes, os.PathLike)) and os.path.normcase(os.path.normpath(os.fsdecode(path))) == os.path.normcase(os.path.normpath(os.devnull)):
                null_opens.append(event); return
        if writing:
            attempts.append(event); raise RuntimeError('Persistent write forbidden during cold inference')
    sys.addaudithook(audit)
    process = psutil.Process(); initial_memory = process.memory_info()._asdict(); started = time.perf_counter()
    try:
        module = importlib.import_module(entry); imported = time.perf_counter()
        stream = module.infer(NoDatasets(), ROOT/'artifacts/next3/canonical_deployment'/folder)
        assert next(stream) is None
        warmed = time.perf_counter(); memory = process.memory_info()._asdict()
        assert set(stream.gi_frame.f_locals) == {'datasets', 'model_directory_path', 'prepared'}
        modules = sorted(sys.modules)
        forbidden = sorted({name.split('.')[0] for name in modules}&{'lightgbm', 'sklearn', 'pandas', 'pyarrow', 'joblib'})
        if args.variant == 'LEAN_PACKED_SHARED': assert not forbidden, forbidden
        assert not attempts
    finally: active[0] = False
    from src.next3.artifacts import atomic_json, digest, utc_now
    atomic_json(output, {'status': 'PASS', 'created_utc': utc_now(), 'variant': args.variant,
        'import_seconds': imported-started, 'first_warmed_none_seconds': warmed-started,
        'initial_memory': initial_memory, 'post_warm_memory': memory,
        'imported_modules': modules, 'imported_training_packages': forbidden,
        'dataset_accesses': 0, 'persistent_writes': 0, 'network_attempts': 0, 'null_device_opens': len(null_opens),
        'script_sha256': digest(Path(__file__)),
        'boundary': 'Fresh process after psutil/stdlib setup; post-warm includes imports, JIT and prepared model; synthetic H only'}, immutable=True)
    print('NEXT3 COLD PROBE PASS', args.variant, round(warmed-started, 3), memory['rss'], flush=True)


if __name__ == '__main__': main()
