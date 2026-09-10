import numpy as np
from src.streaming.inference import write_engineering_baseline,infer
from src.validation.subprocess_protocol import subprocess_protocol_predictions


def test_fresh_process_official_socket_protocol(tmp_path):
    write_engineering_baseline(tmp_path)
    rng=np.random.default_rng(712)
    examples=[(rng.normal(size=100).astype(np.float32),rng.normal(size=16).astype(np.float32)) for _ in range(3)]
    expected=np.asarray(list(infer(iter(examples),tmp_path))[1:],dtype=np.float32)
    actual,timing=subprocess_protocol_predictions(examples,tmp_path)
    np.testing.assert_array_equal(actual,expected)
    assert timing['consumer_exit_code']==0
    assert timing['process_total_seconds']>=timing['process_start_to_ready_seconds']>0
    assert timing['ready_to_end_wire_seconds']>0
