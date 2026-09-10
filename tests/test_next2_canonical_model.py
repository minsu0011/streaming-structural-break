from pathlib import Path
import json
import numpy as np
import pytest
from src.next2.canonical_model import dumps,loads,record,restore
from src.next2.inference import load_artifact
from src.next2.runtime import prepare

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('role',['DISTILLED','PERFORMANCE'])
def test_canonical_model_roundtrip_exact_and_alias_independent(role):
    folder=ROOT/'artifacts/next2/deployment_research'/role
    if not (folder/'model.json').exists():
        pytest.skip('Research artifacts absent')
    model,_=load_artifact(folder)
    encoded=dumps(model)
    clone=loads(encoded)
    assert dumps(clone)==encoded
    # JSON roundtrip destroys Python string/dict/array alias graphs.
    rebuilt=restore(json.loads(json.dumps(record(model))))
    assert dumps(rebuilt)==encoded
    rng=np.random.default_rng(445191)
    h=rng.normal(size=512).astype(np.float32)
    o=rng.standard_t(4,size=137).astype(np.float32)
    a,b=prepare(model).new_state(h),prepare(clone).new_state(h)
    for point in o:
        assert a.predict_one(point)==b.predict_one(point)


def test_independent_raw_entry_has_identical_canonical_predictive_bytes():
    reference=ROOT/'artifacts/next2/deployment_research/PERFORMANCE'
    actual=ROOT/'artifacts/next2/entry_reproduction/PERFORMANCE/raw_entry_v1'
    if not (actual/'model.json').exists():
        pytest.skip('Independent raw entry artifact absent')
    a,_=load_artifact(reference)
    b,_=load_artifact(actual)
    assert dumps(a)==dumps(b)
    assert (reference/'model.joblib').read_bytes()!=(actual/'model.joblib').read_bytes()


def test_canonical_payload_rejects_corruption_and_trailing_data():
    folder=ROOT/'artifacts/next2/deployment_research/DISTILLED'
    if not (folder/'model.json').exists():
        pytest.skip('Research artifact absent')
    model,_=load_artifact(folder)
    content=dumps(model)
    for altered in (content+b'junk',content[:-4],b'bad'+content[3:]):
        with pytest.raises(ValueError):
            loads(altered)
    item=record(model)
    interior=next(k for k,v in enumerate(item['tree']['features']) if v>=0)
    item['tree']['left'][interior]=interior
    with pytest.raises(ValueError):
        restore(item)
