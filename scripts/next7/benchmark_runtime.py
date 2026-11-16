"""Admitted final-audit scalar benchmark with resumable10M-point workloads."""
from pathlib import Path
import argparse
import json
import os
import pickle
import platform
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.next7.worker_environment import environment
os.environ.update(environment())
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT / 'artifacts/next7/numba_cache'))
import numpy as np
import psutil
from src.next7.common import BASE, read_json, save, csv_write, digest, utc_now, activity, note
from src.next7.benchmark_adapter import Factory, CLASSIFIERS, STATES, memory, checkpoint, restore

METHODS = CLASSIFIERS + STATES
CHUNK = 409600
POINTS = 10000000


def frozen():
    protocol = read_json(BASE / 'LATENCY_WORKLOAD_LOCK_NEXT7.json')
    assert protocol['classifier_methods'] == list(CLASSIFIERS) and protocol['state_methods'] == list(STATES)
    assert protocol['direct_throughput']['points_per_classifier_per_workload'] == POINTS
    exported_path = BASE / 'RESEARCH_BUNDLES_EXPORTED.json'
    assert digest(exported_path) == protocol['research_bundle_manifest_sha256']
    fold_receipt = next(r for r in read_json(exported_path)['folds'] if r['fold'] == 0)
    assert digest(BASE / 'research_bundles/fold_0/MANIFEST.json') == fold_receipt['manifest_sha256']
    sources = {n: digest(ROOT / n) for n in ('scripts/next7/benchmark_runtime.py', 'src/next7/benchmark_adapter.py')}
    path = BASE / 'LATENCY_IMPLEMENTATION_LOCK_NEXT7.json'
    if not path.exists():
        tests = BASE / 'audit/BENCHMARK_CHECKPOINT_TESTS_001.xml'
        xml = ET.parse(tests)
        assert len(xml.findall('.//testcase')) == 7 and not xml.findall('.//failure') and not xml.findall('.//error')
        workload_tests = BASE / 'audit/BENCHMARK_WORKLOAD_TESTS_001.xml'
        workload_xml = ET.parse(workload_tests)
        assert len(workload_xml.findall('.//testcase')) == 4 and not workload_xml.findall('.//failure') and not workload_xml.findall('.//error')
        save(path.name, {'status': 'FROZEN', 'created_utc': utc_now(), 'source_sha256': sources,
            'protocol_sha256': digest(BASE / 'LATENCY_WORKLOAD_LOCK_NEXT7.json'), 'test_xml_sha256': digest(tests),
            'workload_count_and_resume_tests_sha256': digest(workload_tests),
            'checkpoint': 'Exact original scalar state arguments, typed resident wrapper reconstructed on resume; all seven methods continuation parity tested.',
            'throughput_timer_includes': 'Scalar API, Python iteration, float conversion and scalar checksum accumulation; no per-point clock call.',
            'throughput_timer_excludes': 'H initialization (separately measured), checkpoint serialization and inter-method rotation.',
            'repeated_series_validation': 'Each segment final scalar equals the original warm path exactly; aggregate checksum compared at1e-11 relative/1e-6 absolute.',
            'state_only_memory': 'Reachable numeric arrays with immutable model arrays excluded as in NEXT6; bounded mutable bytes separately.',
            'cold': 'Each method is a new process with its own previously nonexistent NumBa cache directory.'})
    assert read_json(path)['source_sha256'] == sources
    return protocol


def audit_gate():
    from src.next7.runtime_accounting import audit_admitted
    from src.next7.common import union_seconds
    session = read_json(BASE / 'SESSION_NEXT7.json')
    scheduler = read_json(BASE / 'runtime/SCHEDULER_STATE.json')
    admission = read_json(BASE / 'runtime/FINAL_AUDIT_ADMISSION.json')
    assert audit_admitted(admission, union_seconds(scheduler['intervals']),
        digest(BASE / 'SESSION_NEXT7.json'), session['research_minimum_seconds'])
    for item in read_json(BASE / 'runtime/TASK_QUEUE.json'):
        parent = BASE / 'runtime/queue' / item['id']
        terminal = read_json(parent / 'TERMINAL.json')
        if terminal['status'] != 'COMPLETE':
            resolved = read_json(parent / 'RESOLUTION.json')
            assert resolved['status'] in ('REPAIRED_AND_REPRODUCED', 'EXPECTED_FAILURE_TEST')
            assert resolved['terminal_sha256'] == digest(parent / 'TERMINAL.json')
        for path in parent.glob('attempt_*/START.json'):
            record = read_json(path)
            from src.next7.queue import identity_alive
            assert not identity_alive(record['pid'], record['process_created']), 'Owned research runner is still active'


def workload():
    from src.next2.counterexamples import generate
    return generate('N2', 63, partition=37, horizon=4096, control=True,
        lock=read_json(ROOT / 'COUNTEREXAMPLE_SUITE_LOCK.json'))[:2]


def cold(method):
    frozen()
    audit_gate()
    h, o = workload()
    start = time.perf_counter()
    factory = Factory(method)
    preparation = time.perf_counter() - start
    start = time.perf_counter()
    state = factory.new_state(h)
    initial = time.perf_counter() - start
    start = time.perf_counter()
    value = state.predict_one(float(o[0]))
    first = time.perf_counter() - start
    assert np.isfinite(value).all()
    print(json.dumps({'method': method, 'model_load_prepare_seconds': preparation,
        'cold_H_initialization_seconds': initial, 'first_scalar_seconds': first,
        'new_process': True, 'empty_owned_numba_cache_at_process_start': True,
        'synthetic_generation_before_timer': True, 'output_finite': True}))


def cold_all():
    rows = []
    for method in METHODS:
        destination = BASE / f'latency/cold/{method}.json'
        if destination.exists():
            rows.append(read_json(destination))
            continue
        attempt = uuid.uuid4().hex
        cache = BASE / f'latency/cold/cache_{method}_{attempt}'
        cache.mkdir(parents=True, exist_ok=False)
        assert not any(cache.iterdir())
        env = environment()
        env['NUMBA_CACHE_DIR'] = str(cache)
        start = time.perf_counter()
        run = subprocess.run([sys.executable, '-B', str(Path(__file__)), '--cold-method', method],
            cwd=ROOT, env=env, capture_output=True, text=True, encoding='utf-8',
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        wall = time.perf_counter() - start
        save(f'latency/cold/{method}_{attempt}.log', run.stdout + '\n' + run.stderr)
        assert run.returncode == 0, run.stderr
        result = json.loads(run.stdout)
        result.update({'subprocess_wall_seconds': wall, 'attempt': attempt,
            'implementation_sha256': digest(BASE / 'LATENCY_IMPLEMENTATION_LOCK_NEXT7.json')})
        save(str(destination.relative_to(BASE)), result)
        rows.append(result)
        print(utc_now(), 'cold latency complete', method, flush=True)
    return rows


def write_pickle(path, value):
    path = Path(path).resolve()
    assert path.is_relative_to((BASE / 'latency').resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists()
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temp.open('xb') as f:
        pickle.dump(value, f, protocol=5)
        f.flush()
        os.fsync(f.fileno())
    os.rename(temp, path)


def warm(factories, h, o):
    relative = 'latency/warm'
    complete = BASE / relative / 'DONE.json'
    if complete.exists():
        done = read_json(complete)
        with np.load(BASE / relative / 'PATHS_AND_TIMINGS.npz', allow_pickle=False) as f:
            return done['rows'], {method: f['path_' + method].copy() for method in CLASSIFIERS}
    # All signatures warmed before reporting warm API costs.
    for method in METHODS:
        factories[method].new_state(h).predict_one(float(o[0]))
    timers = {m: [] for m in METHODS}
    blocks = {m: [] for m in METHODS}
    initialization = {m: [] for m in METHODS}
    sizes, paths = {}, {}
    cpu_before = psutil.cpu_percent(interval=.2)
    for repetition in range(7):
        order = METHODS[repetition % len(METHODS):] + METHODS[:repetition % len(METHODS)]
        for method in order:
            start = time.perf_counter_ns()
            state = factories[method].new_state(h)
            initialization[method].append((time.perf_counter_ns() - start) / 1e9)
            local = np.empty(len(o), dtype=np.int64)
            output = np.empty(len(o)) if method in CLASSIFIERS else None
            block_start = time.perf_counter_ns()
            for t, value in enumerate(o):
                start = time.perf_counter_ns()
                result = state.predict_one(float(value))
                local[t] = time.perf_counter_ns() - start
                if output is not None:
                    output[t] = result
            blocks[method].append((time.perf_counter_ns() - block_start) / len(o))
            assert np.isfinite(result).all()
            timers[method].append(local)
            sizes[method] = memory(method, state)
            if output is not None:
                if method in paths:
                    np.testing.assert_array_equal(output, paths[method])
                else:
                    paths[method] = output
    rows = []
    baseline = np.median(np.concatenate(timers['M0']))
    for method in METHODS:
        times = np.concatenate(timers[method])
        q = np.quantile(times, [.5, .95, .99]) / 1000
        rows.append({'method': method, 'kind': 'CLASSIFIER' if method in CLASSIFIERS else 'STATE_DIAGNOSTIC',
            'scalar_calls': len(times), 'median_us': float(q[0]), 'p95_us': float(q[1]), 'p99_us': float(q[2]),
            'median_ratio_vs_M0': float(q[0] * 1000 / baseline),
            'within_3x_M0_median_feasibility': bool(q[0] * 1000 <= 3 * baseline),
            'median_instrumented_block_average_us': float(np.median(blocks[method]) / 1000),
            'warm_H_initialization_median_seconds': float(np.median(initialization[method])),
            'reachable_numeric_array_bytes_excluding_models': sizes[method][0],
            'bounded_mutable_stream_or_added_branch_bytes': sizes[method][1],
            'memory_byte_scope': 'Added NEXT7 branch only' if method in CLASSIFIERS[2:] else 'Whole state when available',
            'state_only_10M_seconds_estimate_instrumented': float(np.median(blocks[method]) * POINTS / 1e9) if method in STATES else None,
            'state_only_direct_10M_executed': False, 'python_object_overhead_excluded': True})
    from scripts.next7.conditional_null_grid import atomic_npz
    atomic_npz(BASE / relative / 'PATHS_AND_TIMINGS.npz',
        **{'path_' + m: p for m, p in paths.items()}, **{'timing_ns_' + m: np.concatenate(v) for m, v in timers.items()})
    csv_write(relative + '/SUMMARY.csv', rows)
    save(relative + '/DONE.json', {'status': 'COMPLETE', 'created_utc': utc_now(), 'rows': rows,
        'system_cpu_percent_before': cpu_before, 'same_method_repeat_output_exact': True,
        'files_sha256': {p.name: digest(p) for p in (BASE / relative).iterdir() if p.is_file()}})
    return rows, paths


def H_initialization(factories):
    destination = BASE / 'latency/H_initialization/DONE.json'
    if destination.exists():
        return read_json(destination)['rows']
    from src.next7.conditional_null import source_components
    h, _, _, _ = source_components('N2', 0, read_json(ROOT / 'COUNTEREXAMPLE_SUITE_LOCK.json'))
    assert len(h) == 5000
    rows = []
    for length in (1000, 1500, 2048, 3000, 5000):
        for method in METHODS:
            times = []
            for repetition in range(3):
                start = time.perf_counter()
                state = factories[method].new_state(h[-length:])
                times.append(time.perf_counter() - start)
            rows.append({'method': method, 'H_length': length, 'repetitions': 3,
                'median_H_initialization_seconds': float(np.median(times)), 'measurements_seconds': times,
                'reachable_numeric_array_bytes_excluding_models': memory(method, state)[0]})
    csv_write('latency/H_initialization/SUMMARY.csv', rows)
    save('latency/H_initialization/DONE.json', {'status': 'COMPLETE', 'created_utc': utc_now(), 'rows': rows})
    return rows


def throughput(workload_name, factories, h, o, warm_paths):
    directory = BASE / 'latency/throughput' / workload_name
    done_path = directory / 'DONE.json'
    if done_path.exists():
        return read_json(done_path)['rows']
    identity = digest(BASE / 'LATENCY_IMPLEMENTATION_LOCK_NEXT7.json')
    progress, states = {}, {}
    for method in CLASSIFIERS:
        existing = sorted((directory / method).glob('chunk_*.pkl'))
        if existing:
            with existing[-1].open('rb') as f:
                stored = pickle.load(f)
            assert stored['identity'] == identity and stored['workload'] == workload_name and stored['method'] == method
            progress[method] = stored['progress']
            if workload_name == 'LONG_SINGLE_STATE':
                states[method] = restore(factories[method], stored['state'])
        else:
            start = time.perf_counter()
            state = factories[method].new_state(h)
            initial = time.perf_counter() - start
            progress[method] = {'points': 0, 'point_seconds': 0., 'H_seconds': initial if workload_name == 'LONG_SINGLE_STATE' else 0.,
                'H_initializations': 1 if workload_name == 'LONG_SINGLE_STATE' else 0,
                'checksum': 0., 'chunks': 0, 'last_value': None, 'first_size': list(memory(method, state)),
                'first_RSS_bytes': psutil.Process().memory_info().rss}
            if workload_name == 'LONG_SINGLE_STATE':
                states[method] = state
    tiled = np.tile(o, CHUNK // len(o))
    rotations = (POINTS + CHUNK - 1) // CHUNK
    for rotation in range(rotations):
        order = CLASSIFIERS[rotation % len(CLASSIFIERS):] + CLASSIFIERS[:rotation % len(CLASSIFIERS)]
        target = min((rotation + 1) * CHUNK, POINTS)
        for method in order:
            record = progress[method]
            if record['points'] >= target:
                continue
            assert record['points'] == rotation * CHUNK
            amount = target - record['points']
            subtotal = 0.
            if workload_name == 'LONG_SINGLE_STATE':
                state = states[method]
                start = time.perf_counter()
                for value in tiled[:amount]:
                    last = state.predict_one(float(value))
                    subtotal += last
                record['point_seconds'] += time.perf_counter() - start
            else:
                processed = 0
                while processed < amount:
                    length = min(len(o), amount - processed)
                    start = time.perf_counter()
                    state = factories[method].new_state(h)
                    record['H_seconds'] += time.perf_counter() - start
                    record['H_initializations'] += 1
                    start = time.perf_counter()
                    for value in o[:length]:
                        last = state.predict_one(float(value))
                        subtotal += last
                    record['point_seconds'] += time.perf_counter() - start
                    assert last == warm_paths[method][length - 1]
                    processed += length
                expected = float(np.sum(warm_paths[method])) * (amount // len(o)) + float(np.sum(warm_paths[method][:amount % len(o)]))
                assert np.isclose(subtotal, expected, rtol=1e-11, atol=1e-6)
            assert np.isfinite(last) and 0 <= last <= 1
            assert list(memory(method, state)) == record['first_size'], 'Numeric state grew with online time'
            record.update(points=target, chunks=rotation + 1, last_value=float(last),
                checksum=record['checksum'] + subtotal, last_RSS_bytes=psutil.Process().memory_info().rss)
            if workload_name == 'LONG_SINGLE_STATE' and method in ('G2_EVENT', 'G2_FULL', 'G3_PAIRWISE_FULL'):
                snapshot = state.snapshot()
                assert snapshot[43] == 0., 'New qualification must be unsupported after16384 points'
                stable = snapshot[[30, 31, 34, 35, 36, 37, 38, 39]].tolist()
                if 'qualified_memory_after_first_chunk' in record:
                    assert record['qualified_memory_after_first_chunk'] == stable
                record['qualified_memory_after_first_chunk'] = stable
            payload = {'identity': identity, 'workload': workload_name, 'method': method,
                'created_utc': utc_now(), 'progress': record,
                'state': checkpoint(method, state) if workload_name == 'LONG_SINGLE_STATE' else None}
            write_pickle(directory / method / f'chunk_{rotation + 1:03d}.pkl', payload)
            print(utc_now(), workload_name, method, target, '/10000000', flush=True)
    rows = []
    for method, record in progress.items():
        assert record['points'] == POINTS
        expected_H = 1 if workload_name == 'LONG_SINGLE_STATE' else (POINTS + len(o) - 1) // len(o)
        assert record['H_initializations'] == expected_H
        rows.append({'workload': workload_name, 'method': method, **record,
            'measured_point_microseconds': record['point_seconds'] * 1e6 / POINTS,
            'measured_online_plus_H_seconds': record['point_seconds'] + record['H_seconds'],
            'direct_10M_points': True, 'numeric_state_size_stable': True,
            'includes_Python_loop_and_checksum': True, 'checkpoint_serialization_excluded_from_throughput_timer': True,
            'periodic_workload_has_no_scientific_detection_claim': True})
    csv_write(str((directory / 'SUMMARY.csv').relative_to(BASE)), rows)
    save(str(done_path.relative_to(BASE)), {'status': 'COMPLETE', 'created_utc': utc_now(),
        'rows': rows, 'total_scalar_calls': POINTS * len(CLASSIFIERS), 'implementation_sha256': identity,
        'checkpoints_sha256': {p.relative_to(BASE).as_posix(): digest(p) for p in directory.glob('*/*.pkl')}})
    return rows


def main():
    frozen()
    audit_gate()
    assert not (BASE / 'LATENCY_NEXT7_DONE.json').exists()
    with activity('Final scalar cold/warm and direct10M workload audit', 'FINAL_AUDIT', 'REPRODUCTION_AND_RUNTIME'):
        cold_rows = cold_all()
        h, o = workload()
        factories = {method: Factory(method) for method in METHODS}
        warm_rows, paths = warm(factories, h, o)
        h_rows = H_initialization(factories)
        direct = []
        for name in ('LONG_SINGLE_STATE', 'REPEATED_FRESH_SERIES'):
            direct += throughput(name, factories, h, o, paths)
        csv_write('LATENCY_NEXT7.csv', warm_rows)
        csv_write('COLD_START_NEXT7.csv', cold_rows)
        csv_write('H_INITIALIZATION_LATENCY_NEXT7.csv', h_rows)
        csv_write('DIRECT_10M_THROUGHPUT_NEXT7.csv', direct)
        save('LATENCY_NEXT7_DONE.json', {'status': 'COMPLETE', 'created_utc': utc_now(),
            'platform': platform.platform(), 'logical_CPUs': psutil.cpu_count(),
            'physical_memory_bytes': psutil.virtual_memory().total, 'warm_rows': warm_rows,
            'direct_calls': POINTS * len(CLASSIFIERS) * 2, 'owned_heavy_jobs_finished_before_benchmark': True,
            'workload_sha256': digest(BASE / 'LATENCY_WORKLOAD_LOCK_NEXT7.json'),
            'implementation_sha256': digest(BASE / 'LATENCY_IMPLEMENTATION_LOCK_NEXT7.json'),
            'final_audit_admission_sha256': digest(BASE / 'runtime/FINAL_AUDIT_ADMISSION.json'),
            'hardware_independent_speed_guarantee': False, 'predictive_promotion_from_speed': False})
    note('Final scalar runtime audit complete', 'Both10M workloads and separate initialization/cold costs recorded',
        'FINAL_AUDIT', phase='FINAL_AUDIT')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lock-only', action='store_true')
    parser.add_argument('--cold-method', choices=METHODS)
    args = parser.parse_args()
    if args.lock_only:
        frozen()
    elif args.cold_method:
        cold(args.cold_method)
    else:
        main()
