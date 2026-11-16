"""Fresh-process write denial starts before importing the chosen project entry."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import importlib
import json
import math
import os
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class SinglePass:
    def __init__(self, values):
        self.values = iter(values)
        self.reads = 0
        self.entered = False
    def __iter__(self):
        if self.entered:
            raise AssertionError('Input iterator traversed twice')
        self.entered = True
        return self
    def __next__(self):
        value = next(self.values)
        self.reads += 1
        return value
    def __len__(self):
        raise AssertionError('Online length inspected')
    def __getitem__(self, key):
        raise AssertionError('Online stream indexed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entry', required=True)
    parser.add_argument('--fixture', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--expect-import-write', action='store_true')
    args = parser.parse_args()
    fixture = json.loads(Path(args.fixture).read_text(encoding='utf-8'))
    active = [True]
    events = {'read_opens': 0, 'filesystem_mutation_attempts': []}
    def audit(event, arguments):
        if not active[0]:
            return
        mutation = event in ('os.mkdir', 'os.remove', 'os.rename', 'os.rmdir', 'os.chmod', 'os.truncate', 'os.link', 'os.symlink')
        if event == 'open':
            path, mode, flags = arguments
            mutation = (isinstance(mode, str) and any(char in mode for char in 'wax+')) or (isinstance(flags, int) and bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
            if not mutation:
                events['read_opens'] += 1
        if mutation:
            events['filesystem_mutation_attempts'].append({'event': event, 'path': str(arguments[0])})
            raise RuntimeError('COLD_IMPORT_WRITE_DENIED')
    sys.addaudithook(audit)
    started = time.perf_counter()
    try:
        entry = importlib.import_module(args.entry)
        if args.expect_import_write:
            raise AssertionError('The negative control unexpectedly imported without a filesystem mutation')
        import numpy as np
        online = [SinglePass(series['online']) for series in fixture['series']]
        outer = SinglePass([(series['historical'], stream) for series, stream in zip(fixture['series'], online)])
        generator = entry.infer(outer, fixture['model_directory'])
        assert next(generator) is None
        assert outer.reads == 0 and all(stream.reads == 0 for stream in online)
        count = 0
        for j, series in enumerate(fixture['series']):
            for t, expected in enumerate(series['expected']):
                actual = next(generator)
                assert isinstance(actual, float) and math.isfinite(actual)
                assert np.float32(actual) == np.float32(expected)
                assert outer.reads == j+1 and online[j].reads == t+1
                assert all(stream.reads == 0 for stream in online[j+1:])
                count += 1
        try:
            next(generator)
            raise AssertionError('Inference emitted more predictions than inputs')
        except StopIteration:
            pass
        from src.next.readonly_bootstrap import cache_requests_suppressed, source_hash
        record = {'status': 'PASS', 'entry': args.entry, 'cold_import_included_in_write_guard': True,
            'first_jit_readiness_included_in_write_guard': True, 'points': count, 'series': len(online),
            'bitwise_oof_parity': True, 'readiness_consumes_zero_inputs': True,
            'single_pass_and_no_read_ahead': True, 'bootstrap_sha256': source_hash(),
            'project_cache_initializations_suppressed': len(cache_requests_suppressed()), 'audit': events}
    except RuntimeError as exc:
        if not args.expect_import_write or str(exc) != 'COLD_IMPORT_WRITE_DENIED':
            raise
        record = {'status': 'PASS_EXPECTED_IMPORT_WRITE_REPRODUCED', 'entry': args.entry,
            'reason': 'The old NEXT entry enables a project Numba cache locator during module import, before its later cache disabling call.',
            'actual_filesystem_mutations_performed': 0, 'audit': events}
    finally:
        active[0] = False
    record['elapsed_seconds'] = time.perf_counter()-started
    Path(args.output).write_text(json.dumps(record, indent=2, allow_nan=False), encoding='utf-8')
    print('COLD IMPORT CHECK', args.entry, record['status'], flush=True)


if __name__ == '__main__':
    main()
