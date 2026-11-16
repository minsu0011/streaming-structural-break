"""Frozen E003 public outputs against every saved original long-null/O1 score."""
import os
import sys
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import csv
import hashlib
import importlib
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Public bootstrap precedes all project numerical/training-module imports.
entry = importlib.import_module('next3_lean_performance_main')
import numpy as np
from src.next3.artifacts import BASE, atomic_json, digest, progress, read_json, utc_now
from src.next3.portable_probes import Online
from src.next2.counterexamples import generate, policy, source_hash as generator_hash

BL = 'BL2_C205_DEPENDENCE_80'
CASES = [('long_null', name, 512, 16384, 2) for name in ('N0', 'N1', 'N2', 'N3')]
CASES.append(('counterexample', 'O1', 1024, 1024, 1))


class ScenarioDatasets:
    def __init__(self, scenario, count, horizon, partition, suite):
        self.scenario = scenario; self.count = count; self.horizon = horizon
        self.partition = partition; self.suite = suite; self.accessed = 0; self.current = None
        self.historical = None; self.input_sha = None

    def __iter__(self): return self

    def __next__(self):
        if self.current is not None:
            assert self.current.consumed == self.horizon
            assert hashlib.sha256(self.historical.tobytes()+self.current.values.tobytes()).hexdigest() == self.input_sha
        if self.accessed == self.count: raise StopIteration
        row = self.accessed
        control = self.scenario == 'O1' and row < 512
        historical, online, synthetic_tau = generate(self.scenario, row % 512,
            partition=self.partition, horizon=self.horizon, control=control, lock=self.suite)
        assert synthetic_tau == -1 and len(historical) == 2048 and len(online) == self.horizon
        self.historical = historical; self.current = Online(online)
        self.input_sha = hashlib.sha256(historical.tobytes()+online.tobytes()).hexdigest()
        self.accessed += 1
        return self.historical, self.current

    def __len__(self): raise AssertionError('Public inference inspected future dataset count')
    def __length_hint__(self): raise AssertionError('Public inference inspected future dataset count')


def main():
    protocol_path = BASE/'maintenance/LEAN_LONG_HORIZON_PARITY_PROTOCOL.json'
    protocol = read_json(protocol_path)
    assert protocol['status'] == 'REGISTERED_ZERO_FIT_RUNTIME_PARITY'
    for name, sha in protocol['source_sha256'].items(): assert digest(ROOT/name) == sha
    out = BASE/'maintenance/lean_long_horizon_parity_v1'; out.mkdir(parents=True, exist_ok=False)
    suite = policy(); assert digest(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json') == protocol['counterexample_lock_sha256']
    assert generator_hash() == protocol['generator_sha256']
    model = BASE/'canonical_deployment/PERFORMANCE_lean_shared_v1'
    assert digest(model/'model.sbnext2') == protocol['canonical_model_sha256']
    evidence = {}; expected_arrays = {}; per_case = []; summaries = []; points = 0; began = time.perf_counter()
    for kind, scenario, count, horizon, partition in CASES:
        directory = ROOT/'artifacts/next2/finalist_counterexamples'/kind/scenario
        receipt_path = directory/'RESULTS.json'; receipt = read_json(receipt_path)
        assert receipt['status'] == 'PASS' and receipt['identity']['counterexample_lock_sha256'] == protocol['counterexample_lock_sha256']
        array_path = directory/(BL+'.npy')
        assert digest(array_path) == receipt['files_sha256'][array_path.name]
        for path in (receipt_path, array_path): evidence[path.relative_to(ROOT).as_posix()] = digest(path)
        array = np.load(array_path, mmap_mode='r', allow_pickle=False)
        assert array.dtype == np.float32 and array.shape == (count, horizon)
        expected_arrays[scenario] = array
    atomic_json(out/'START.json', {'created_utc': utc_now(), 'protocol_sha256': digest(protocol_path),
        'existing_synthetic_score_files_sha256': evidence, 'expected_public_points': 34603008,
        'new_fits': 0, 'new_quality_scores': 0, 'new_runtime_families': 0}, immutable=True)
    for kind, scenario, count, horizon, partition in CASES:
        datasets = ScenarioDatasets(scenario, count, horizon, partition, suite)
        stream = entry.infer(datasets, model)
        assert next(stream) is None and datasets.accessed == 0
        fingerprint = hashlib.sha256(); scenario_started = time.perf_counter()
        for row in range(count):
            output = np.empty(horizon, dtype=np.float32)
            for index in range(horizon):
                value = next(stream)
                assert type(value) is float and np.isfinite(value) and 0. <= value <= 1.
                assert datasets.accessed == row+1 and datasets.current.consumed == index+1
                output[index] = value
            np.testing.assert_array_equal(output, expected_arrays[scenario][row])
            state_bytes = stream.gi_frame.f_locals['state'].state_array_bytes
            assert state_bytes == 16925
            content = output.tobytes(); fingerprint.update(content); points += horizon
            per_case.append({'scenario': scenario, 'kind': kind, 'synthetic_row_index': row,
                'historical_points': 2048, 'online_points': horizon, 'input_sha256': datasets.input_sha,
                'float32_prediction_sha256': hashlib.sha256(content).hexdigest(),
                'matches_original_saved_output': True, 'retained_numeric_bytes': state_bytes})
            if (row+1) % 64 == 0:
                progress('E003 long-horizon public compatibility',
                    f'{scenario}: {row+1}/{count} original synthetic streams exact; {points} total public outputs; no new quality metric')
                print('NEXT3 LEAN LONG HORIZON', scenario, row+1, count, points, flush=True)
        try: next(stream)
        except StopIteration: pass
        else: raise AssertionError('Public inference yielded extra synthetic output')
        assert datasets.accessed == count
        expected_sha = hashlib.sha256(expected_arrays[scenario].tobytes()).hexdigest()
        assert fingerprint.hexdigest() == expected_sha
        summaries.append({'kind': kind, 'scenario': scenario, 'series': count, 'horizon': horizon,
            'public_points': count*horizon, 'float32_output_sha256': fingerprint.hexdigest(),
            'original_float32_output_sha256': expected_sha, 'max_abs_difference': 0.,
            'retained_numeric_bytes_every_series_end': 16925,
            'verification_wall_seconds_not_latency_benchmark': time.perf_counter()-scenario_started})
    assert points == 34603008 and len(per_case) == 3072
    for name, sha in evidence.items(): assert digest(ROOT/name) == sha
    for name, sha in protocol['source_sha256'].items(): assert digest(ROOT/name) == sha
    with (out/'PER_SYNTHETIC_SERIES.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_case[0])); writer.writeheader(); writer.writerows(per_case)
    atomic_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(), 'scenarios': summaries,
        'public_API_prediction_points': points, 'synthetic_streams': len(per_case),
        'all_original_float32_predictions_exact': True, 'input_arrays_unchanged': True,
        'first_None_before_dataset_access': True, 'one_current_point_per_yield': True,
        'constant_retained_numeric_bytes': 16925, 'new_fits': 0, 'new_quality_scores': 0,
        'new_runtime_families': 0, 'real_DEV_or_seal_values_accessed': False,
        'thresholds_recalibrated': False, 'new_alarm_or_quality_metrics_computed': False,
        'existing_synthetic_score_files_sha256': evidence, 'protocol_sha256': digest(protocol_path),
        'verification_wall_seconds_not_latency_benchmark': time.perf_counter()-began,
        'files_sha256': {path.name: digest(path) for path in out.iterdir() if path.is_file()},
        'scope': 'Exact compatibility of the frozen lean public entry with previously saved float32 outputs on all original N0-N3 16384-point streams and original O1 paired controls. Does not repair false alarms, add quality evidence, retune thresholds or replace final raw/30M/archive verification.'}, immutable=True)
    progress('E003 original long-horizon compatibility complete',
        '34,603,008 public outputs across 3,072 old synthetic streams exact; 16,925 retained numeric bytes; no fits or new quality metrics')
    print('NEXT3 LEAN LONG-HORIZON COMPATIBILITY PASS', points, flush=True)


if __name__ == '__main__': main()
