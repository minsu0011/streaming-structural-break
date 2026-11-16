"""Bounded public-generator timing probe with exact current-point accounting."""
import hashlib
import math
import time
import numpy as np


class RepeatedOnline:
    def __init__(self, buffer, points):
        self._buffer = np.asarray(buffer, dtype=np.float32)
        if self._buffer.ndim != 1 or len(self._buffer) == 0 or not np.isfinite(self._buffer).all():
            raise ValueError('Finite nonempty one-dimensional fixed buffer required')
        if type(points) is not int or points <= 0: raise ValueError('Positive fixed point count required')
        self._points = points; self.consumed = 0; self._iterated = False

    def __iter__(self):
        if self._iterated: raise AssertionError('Online iterator consumed twice')
        self._iterated = True; return self

    def __next__(self):
        if self.consumed == self._points: raise StopIteration
        value = self._buffer[self.consumed % len(self._buffer)]; self.consumed += 1; return value

    def __len__(self): raise AssertionError('Future online length requested')
    def __length_hint__(self): raise AssertionError('Future online length hint requested')
    def __array__(self, *args, **kwargs): raise AssertionError('Future online values materialized')


class OneDataset:
    def __init__(self, historical, online):
        self._historical = historical; self.online = online; self.accessed = 0; self._iterated = False

    def __iter__(self):
        if self._iterated: raise AssertionError('Dataset iterator consumed twice')
        self._iterated = True; return self

    def __next__(self):
        if self.accessed: raise StopIteration
        self.accessed += 1; return self._historical, self.online

    def __len__(self): raise AssertionError('Dataset count requested')
    def __length_hint__(self): raise AssertionError('Dataset count hint requested')


def consume_public_probe(entry, directory, historical, buffer, *, points, block_points, on_block=None):
    if type(block_points) is not int or block_points <= 0 or points % block_points:
        raise ValueError('Exact fixed block partition required')
    online = RepeatedOnline(buffer, points); datasets = OneDataset(historical, online)
    before_inputs = (hashlib.sha256(np.asarray(historical).tobytes()).hexdigest(),
                     hashlib.sha256(np.asarray(buffer).tobytes()).hexdigest())
    start = time.perf_counter(); stream = entry.infer(datasets, directory)
    if next(stream) is not None or datasets.accessed != 0:
        raise AssertionError('Initial None must precede all dataset access')
    first_none = time.perf_counter()-start
    values = np.empty(block_points, dtype=np.float32); fingerprint = hashlib.sha256()
    records = []; measured = 0.; completed = 0; initial_bytes = None; final_bytes = None
    wall = time.perf_counter()
    while completed < points:
        tick = time.perf_counter_ns()
        for index in range(block_points):
            value = next(stream)
            if (type(value) is not float or not math.isfinite(value) or not 0. <= value <= 1.
                    or online.consumed != completed+index+1):
                raise AssertionError('Exactly one current observation per Python-float prediction required')
            values[index] = value
            if initial_bytes is None:
                state = stream.gi_frame.f_locals['state']
                initial_bytes = int(state.state_array_bytes)
        duration = (time.perf_counter_ns()-tick)/1e9; measured += duration; completed += block_points
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise AssertionError('Finite [0,1] outputs required')
        fingerprint.update(values.tobytes()); state = stream.gi_frame.f_locals['state']
        final_bytes = int(state.state_array_bytes)
        if final_bytes != initial_bytes: raise AssertionError('Retained numeric state grew during the stream')
        record = {'completed_points': completed, 'generator_loop_seconds': duration,
            'cumulative_generator_loop_seconds': measured, 'state_array_bytes': final_bytes}
        records.append(record)
        if on_block is not None: on_block(record)
    sentinel = object()
    if next(stream, sentinel) is not sentinel or online.consumed != points or datasets.accessed != 1:
        raise AssertionError('Public generator did not terminate exactly after its only series')
    assert before_inputs == (hashlib.sha256(np.asarray(historical).tobytes()).hexdigest(),
                             hashlib.sha256(np.asarray(buffer).tobytes()).hexdigest())
    return {'points': completed, 'output_sha256': fingerprint.hexdigest(),
        'measured_generator_loop_seconds': measured, 'wall_seconds_including_checks': time.perf_counter()-wall,
        'ready_None_seconds_in_existing_process': first_none, 'state_array_bytes_after_first_point': initial_bytes,
        'state_array_bytes_after_last_point': final_bytes, 'first_none_before_data': True,
        'one_current_point_per_yield': True, 'inputs_unchanged': True, 'blocks': records,
        'timing_scope': 'Actual public generator loop, including current-point accounting and one real H initialization. Excludes initial ready-None warm-up; not a pure-kernel or cold-process measurement.'}
