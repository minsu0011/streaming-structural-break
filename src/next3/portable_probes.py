"""Raw-free deterministic public-API probes with no accessible future iterator API."""
import hashlib
import numpy as np

SEED = 2026091081
HISTORICAL_LENGTHS = (128, 257, 512, 2048)
SCENARIOS = ('gaussian', 'single_outlier', 'input_scale_down', 'dependence_flip')
ONLINE_LENGTH = 1537


def cases():
    result = []
    for hi, historical_length in enumerate(HISTORICAL_LENGTHS):
        for si, scenario in enumerate(SCENARIOS):
            rng = np.random.default_rng(np.random.SeedSequence([SEED, hi, si]))
            noise = rng.normal(size=128+historical_length+ONLINE_LENGTH)
            values = np.empty(len(noise), dtype=np.float64); previous = 0.
            for j, value in enumerate(noise):
                phi = -.4 if scenario == 'dependence_flip' and j >= 128+historical_length+512 else .4
                previous = phi*previous+np.sqrt(1.-phi*phi)*value; values[j] = previous
            historical = values[128:128+historical_length].astype(np.float32)
            online = values[128+historical_length:].astype(np.float32)
            if scenario == 'single_outlier': online[37] += 20 if hi%2 == 0 else -20
            if scenario == 'input_scale_down': online[512:] *= np.float32(.7)
            result.append((f'H{historical_length}_{scenario}', historical, online))
    # Empty and one-point iterators exercise termination boundaries without
    # changing the longer scientific synthetic cohort or introducing labels.
    _, historical, online = result[0]
    result.extend([('EMPTY_ONLINE', historical.copy(), np.empty(0, dtype=np.float32)),
                   ('ONE_POINT', historical.copy(), online[:1].copy())])
    return result


class Online:
    def __init__(self, values): self.values = values; self.consumed = 0; self.iterated = False
    def __iter__(self):
        if self.iterated: raise AssertionError('Synthetic online iterator consumed twice')
        self.iterated = True; return self
    def __next__(self):
        if self.consumed == len(self.values): raise StopIteration
        result = self.values[self.consumed]; self.consumed += 1; return result
    def __len__(self): raise AssertionError('Online future length accessed')
    def __length_hint__(self): raise AssertionError('Online future length hint accessed')
    def __array__(self, *args, **kwargs): raise AssertionError('Online suffix materialized')


class Datasets:
    def __init__(self, values): self.values = values; self.accessed = 0; self.current = None; self.case_index = -1
    def __iter__(self): return self
    def __next__(self):
        if self.current is not None: assert self.current.consumed == len(self.current.values)
        if self.accessed == len(self.values): raise StopIteration
        self.case_index = self.accessed; _, historical, online = self.values[self.accessed]
        self.accessed += 1; self.current = Online(online); return historical, self.current
    def __len__(self): raise AssertionError('Dataset future length accessed')
    def __length_hint__(self): raise AssertionError('Dataset future length hint accessed')


def replay(entry, directory, values):
    datasets = Datasets(values); stream = entry.infer(datasets, directory)
    assert next(stream) is None and datasets.accessed == 0
    fingerprint = hashlib.sha256(); records = []; points = 0
    # Empty streams emit no score; the next non-empty call advances over them.
    for index, (name, historical, online) in enumerate(values):
        prediction = np.empty(len(online), dtype=np.float32)
        for j in range(len(online)):
            value = next(stream)
            assert type(value) is float and np.isfinite(value) and 0 <= value <= 1
            assert datasets.case_index == index and datasets.current.consumed == j+1
            prediction[j] = value
        content = prediction.tobytes(); fingerprint.update(content); points += len(prediction)
        records.append({'case': name, 'historical_points': len(historical), 'online_points': len(online),
            'input_sha256': hashlib.sha256(historical.tobytes()+online.tobytes()).hexdigest(),
            'prediction_sha256': hashlib.sha256(content).hexdigest()})
    try: next(stream)
    except StopIteration: pass
    else: raise AssertionError('Extra public inference output after all synthetic cases')
    assert datasets.accessed == len(values)
    return {'cases': records, 'points': points, 'output_fingerprint_sha256': fingerprint.hexdigest(),
        'first_none_before_data': True, 'one_current_point_per_output': True, 'no_future_iterator_API_access': True}
