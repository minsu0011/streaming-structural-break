"""Small independent call-count and interruption checks, never timing claims."""
import numpy as np
import pytest
from src.next7.common import atomic
from scripts.next7 import benchmark_runtime as benchmark


class ProbeState:
    def __init__(self):
        self.total = 0.

    def predict_one(self, value):
        self.total += value / 1000
        return self.total


class ProbeFactory:
    def new_state(self, h):
        return ProbeState()


@pytest.mark.parametrize('workload', ['LONG_SINGLE_STATE', 'REPEATED_FRESH_SERIES'])
@pytest.mark.parametrize('interrupted', [False, True])
def test_throughput_counts_partial_series_and_resumes_saved_chunk(tmp_path, monkeypatch, workload, interrupted):
    monkeypatch.setattr(benchmark, 'BASE', tmp_path)
    monkeypatch.setattr(benchmark, 'CLASSIFIERS', ('M0',))
    monkeypatch.setattr(benchmark, 'POINTS', 19)
    monkeypatch.setattr(benchmark, 'CHUNK', 8)
    monkeypatch.setattr(benchmark, 'save', lambda path, value: atomic(tmp_path / path, value))
    monkeypatch.setattr(benchmark, 'csv_write', lambda path, rows: atomic(tmp_path / path, str(rows)))
    monkeypatch.setattr(benchmark, 'memory', lambda method, state: (8, 8))
    monkeypatch.setattr(benchmark, 'checkpoint', lambda method, state: {'method': method, 'total': state.total})
    def restored(factory, payload):
        state = ProbeState()
        state.total = payload['total']
        return state
    monkeypatch.setattr(benchmark, 'restore', restored)
    atomic(tmp_path / 'LATENCY_IMPLEMENTATION_LOCK_NEXT7.json', {'scope': 'unit test fixture'})
    original_write = benchmark.write_pickle
    def fail_after_saved_first_chunk(path, value):
        original_write(path, value)
        raise RuntimeError('Injected interruption after atomic checkpoint')
    h, o = np.zeros(8), np.arange(1, 5, dtype=np.float32)
    warm = np.cumsum(o.astype(float) / 1000)
    if interrupted:
        monkeypatch.setattr(benchmark, 'write_pickle', fail_after_saved_first_chunk)
        with pytest.raises(RuntimeError, match='Injected interruption'):
            benchmark.throughput(workload, {'M0': ProbeFactory()}, h, o, {'M0': warm})
        monkeypatch.setattr(benchmark, 'write_pickle', original_write)
    result = benchmark.throughput(workload, {'M0': ProbeFactory()}, h, o, {'M0': warm})[0]
    assert result['points'] == 19 and result['chunks'] == 3
    if workload == 'LONG_SINGLE_STATE':
        expected = np.cumsum(np.tile(o.astype(float), 5)[:19] / 1000)
        assert result['H_initializations'] == 1
    else:
        expected = np.r_[np.tile(warm, 4), warm[:3]]
        assert result['H_initializations'] == 5
    assert np.isclose(result['checksum'], expected.sum(), rtol=0, atol=1e-15)
    assert result['last_value'] == expected[-1]
    assert len(list((tmp_path / 'latency/throughput' / workload / 'M0').glob('chunk_*.pkl'))) == 3
