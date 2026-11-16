import hashlib
from types import SimpleNamespace
import numpy as np
import pytest
from src.next3.final_runtime_probe import consume_public_probe


def infer(datasets, directory):
    yield None
    for historical, online in datasets:
        state = SimpleNamespace(state_array_bytes=8)
        for value in online: yield float(value)


def test_fixed_buffer_repetition_is_exact_single_pass_and_complete():
    buffer = np.array([.1, .2, .3], dtype=np.float32)
    result = consume_public_probe(SimpleNamespace(infer=infer), None, np.array([0.]), buffer,
                                  points=12, block_points=4)
    assert result['output_sha256'] == hashlib.sha256(np.tile(buffer, 4).tobytes()).hexdigest()
    assert result['points'] == 12 and len(result['blocks']) == 3
    assert result['state_array_bytes_after_last_point'] == result['state_array_bytes_after_first_point'] == 8


@pytest.mark.parametrize('mode', ['early_dataset', 'lookahead', 'length', 'extra_prediction', 'state_growth',
                                  'probability_rounding_escape', 'nonfinite'])
def test_probe_rejects_protocol_violations(mode):
    def bad(datasets, directory):
        if mode == 'early_dataset': next(iter(datasets))
        yield None
        for historical, online in datasets:
            state = SimpleNamespace(state_array_bytes=8)
            if mode == 'length': len(online)
            for value in online:
                if mode == 'lookahead': next(online)
                if mode == 'state_growth': state.state_array_bytes += 1
                if mode == 'probability_rounding_escape': value = 1.+1e-10
                if mode == 'nonfinite': value = float('nan')
                yield float(value)
        if mode == 'extra_prediction': yield .5
    with pytest.raises((AssertionError, StopIteration, RuntimeError)):
        consume_public_probe(SimpleNamespace(infer=bad), None, np.array([0.]), np.array([.1, .2]),
                             points=8, block_points=2)
