import numpy as np
from src.streaming.inference import write_engineering_baseline,infer
from src.validation.protocol import protocol_predictions
from src.validation.synthetic import fixture

def test_official_socket_protocol(tmp_path):
    write_engineering_baseline(tmp_path)
    series=[fixture(kind)[:2] for kind in ['no_break','mean_up','ar_reverse']]
    output=protocol_predictions(infer,series,tmp_path)
    direct=np.asarray(list(infer(iter(series),tmp_path))[1:],dtype=np.float32)
    np.testing.assert_array_equal(output,direct)
