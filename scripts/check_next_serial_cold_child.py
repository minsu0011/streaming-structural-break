"""Fresh-process cold-import/read-only contract against declared expected bytes."""
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
        self.values, self.reads, self.entered = iter(values), 0, False
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
    def __array__(self, *args, **kwargs):
        raise AssertionError('Online stream materialized')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixture', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    fixture = json.loads(Path(args.fixture).read_text(encoding='utf-8'))
    if fixture['prediction_reference_kind'] != 'ALL_DEV_BATCH_FINGERPRINT_NOT_QUALITY_SCORE':
        raise RuntimeError('Explicit all-DEV fingerprint authority required')
    active, events = [True], {'read_opens': 0, 'filesystem_mutation_attempts': []}
    def audit(event, arguments):
        if not active[0]:
            return
        mutation = event in ('os.mkdir', 'os.remove', 'os.rename', 'os.rmdir', 'os.chmod', 'os.truncate', 'os.link', 'os.symlink')
        if event == 'open':
            path, mode, flags = arguments
            mutation = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (isinstance(flags, int) and bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
            if not mutation:
                events['read_opens'] += 1
        if mutation:
            events['filesystem_mutation_attempts'].append({'event': event, 'path': str(arguments[0])})
            raise RuntimeError('COLD_IMPORT_WRITE_DENIED')
    sys.addaudithook(audit)
    tick = time.perf_counter()
    try:
        entry = importlib.import_module('next_serial_main')
        import numpy as np
        streams = [SinglePass(item['online']) for item in fixture['series']]
        outer = SinglePass([(item['historical'], stream) for item, stream in zip(fixture['series'], streams)])
        generator = entry.infer(outer, fixture['model_directory'])
        assert next(generator) is None and outer.reads == 0 and all(s.reads == 0 for s in streams)
        count = 0
        for j, item in enumerate(fixture['series']):
            for t, expected in enumerate(item['expected']):
                actual = next(generator)
                assert type(actual) is float and math.isfinite(actual) and 0. <= actual <= 1.
                assert np.float32(actual) == np.float32(expected)
                assert outer.reads == j+1 and streams[j].reads == t+1
                assert all(s.reads == 0 for s in streams[j+1:])
                count += 1
        try:
            next(generator)
            raise AssertionError('More outputs than observations')
        except StopIteration:
            pass
        from src.next.readonly_bootstrap import cache_requests_suppressed, source_hash
        report = {'status': 'PASS', 'entry': 'next_serial_main', 'cold_import_included_in_write_guard': True,
            'first_jit_readiness_included_in_write_guard': True, 'points': count, 'series': len(streams),
            'bitwise_expected_prediction_parity': True, 'prediction_reference_kind': fixture['prediction_reference_kind'],
            'readiness_consumes_zero_inputs': True, 'single_pass_and_no_read_ahead': True,
            'bootstrap_sha256': source_hash(), 'project_cache_initializations_suppressed': len(cache_requests_suppressed()),
            'model_sha256': fixture['model_sha256'], 'audit': events}
    finally:
        active[0] = False
    report['elapsed_seconds'] = time.perf_counter()-tick
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print('ALL-DEV COLD IMPORT PASS', count, flush=True)


if __name__ == '__main__':
    main()
