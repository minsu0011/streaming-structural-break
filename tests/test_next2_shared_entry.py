from pathlib import Path
import json
import shutil
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT/'artifacts/next2/canonical_deployment/PERFORMANCE_shared_v1'


def test_shared_public_entry_causal_reset_and_saved_reference():
    if not (ARTIFACT/'model.json').exists(): pytest.skip('Shared delivery absent')
    import next2_shared_performance_main as entry
    import next2_canonical_performance_main as reference
    rng = np.random.default_rng(2026090971)
    h = rng.standard_t(5, 1024).astype(np.float32)
    o = rng.standard_t(3, 777).astype(np.float32)
    consumed = []
    def points():
        for i, value in enumerate(o):
            consumed.append(i)
            yield value
    def datasets():
        consumed.append('series')
        yield h, points()
    stream = entry.infer(datasets(), ARTIFACT)
    assert next(stream) is None and consumed == []
    actual = []
    for i in range(len(o)):
        actual.append(next(stream))
        assert consumed == ['series']+list(range(i+1))
    with pytest.raises(StopIteration): next(stream)
    expected = list(reference.infer([(h, iter(o))], ROOT/'artifacts/next2/canonical_deployment/PERFORMANCE_compact_v1'))[1:]
    np.testing.assert_array_equal(actual, expected)
    repeated = list(entry.infer([(h, iter(o[:129])), (h, iter(o))], ARTIFACT))
    np.testing.assert_array_equal(repeated[1:130], actual[:129])
    np.testing.assert_array_equal(repeated[130:], actual)


@pytest.mark.parametrize('mutation', ['runtime_hash', 'model_bytes', 'weight'])
def test_shared_descriptor_rejects_mutation(tmp_path, mutation):
    if not (ARTIFACT/'model.json').exists(): pytest.skip('Shared delivery absent')
    from src.next2.shared_inference import load_artifact
    for name in ('model.json', 'model.sbnext2'): shutil.copyfile(ARTIFACT/name, tmp_path/name)
    descriptor = json.loads((tmp_path/'model.json').read_text())
    if mutation == 'runtime_hash': descriptor['runtime_sha256'] = '0'*64
    elif mutation == 'weight': descriptor['specification']['weight'] = .67
    else:
        path = tmp_path/'model.sbnext2'; data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
    (tmp_path/'model.json').write_text(json.dumps(descriptor))
    with pytest.raises(RuntimeError): load_artifact(tmp_path)
