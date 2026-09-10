import json
from pathlib import Path
import numpy as np
import pytest
from src.next2.used_feature_runtime import UsedOldState,used_arch_step,PreparedUsedPredictor
from src.next.fast_variance_initialization import FastARCHFeatureState
from src.next2.pruned_runtime import PreparedCompactPredictor
from src.next2.io import model_class

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('seed',[43,179,2249])
@pytest.mark.parametrize('columns',[list(range(51)),[5,11,12,24,25,29,30,38,39,43,45,49,50],[0,2,3,4,5]])
def test_sparse_old_features_match_all_original_operations(seed,columns):
    rng=np.random.default_rng(seed)
    h=rng.standard_t(4,512).astype(np.float32)
    o=rng.standard_t(3,193).astype(np.float32)
    original=FastARCHFeatureState(h)
    used=UsedOldState(h,columns)
    for p in o:
        expected=original.update_and_get(p)[columns]
        actual=used_arch_step(float(p),used.call_args)
        np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize('eid',['J102_SERIAL_VARIANCE_512','Y001_LONG_VARIANCE_ARCH1','B104_LONG_VARIANCE_NO_GLR_MEMORY'])
def test_frozen_tree_used_runtime_exact(eid):
    folder=ROOT/'artifacts/next/refits'/eid/'cache_research_v1'
    if not (folder/'model.joblib').exists():
        pytest.skip('Research all-DEV model absent')
    refs=json.loads((ROOT/'artifacts/next2/initialization/FROZEN_REFERENCES.json').read_text())
    ref=next(r for r in refs['references'].values() if r['experiment_id']==eid)
    model=model_class(ref['candidate']).load(folder/'model.joblib')
    rng=np.random.default_rng(75219)
    h=rng.normal(size=640).astype(np.float32)
    o=rng.normal(size=513).astype(np.float32)
    a=PreparedCompactPredictor(model).new_state(h)
    prepared=PreparedUsedPredictor(model)
    b=prepared.new_state(h)
    for point in o:
        expected=a.predict_one(point)
        actual=b.predict_one(point)
        np.testing.assert_array_equal(a.output[prepared.used],b.output)
        assert expected==actual
    assert len(prepared.names)<len(model.names)
