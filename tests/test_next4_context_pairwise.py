import numpy as np
import pytest
from src.next4 import context_pairwise as c
from src.next4.historical_context import context


def test_one_hundred_H_context_and_score_prefix_cases():
    rng=np.random.default_rng(20260908)
    for case in range(100):
        h=rng.standard_t(5,size=128).astype(np.float32);ctx=context(h)
        scores=rng.uniform(size=(48,2));altered=scores.copy();prefix=1+case%32;altered[prefix:]=rng.uniform(size=(48-prefix,2))
        contexts=np.tile(ctx,(48,1));a=c.features(scores,contexts);b=c.features(altered,contexts)
        np.testing.assert_array_equal(a[:prefix],b[:prefix])
        # Readout arithmetic has a fixed column order in batch and scalar use.
        w=rng.normal(size=12)
        np.testing.assert_array_equal(c.pairwise.raw_readout(a,w),np.array([c.pairwise.raw_readout(row[None,:],w)[0] for row in a]))


def test_H_modulation_vanishes_when_experts_agree():
    rng=np.random.default_rng(18);p=rng.uniform(size=100);scores=np.column_stack([p,p]);ctx=rng.normal(size=(100,10))
    np.testing.assert_array_equal(c.features(scores,ctx)[:,2:],np.zeros((100,10)))


def test_context_row_coordinates_reject_id_shuffle_or_missing():
    ids=np.array([2,4,7]);values=np.arange(30).reshape(3,10)
    np.testing.assert_array_equal(c.row_context(ids,values,np.array([7,2,7])),values[[2,0,2]])
    with pytest.raises(ValueError):c.row_context(ids[::-1],values,np.array([2]))
    with pytest.raises(ValueError):c.row_context(ids,values,np.array([3]))


def test_synthetic_nested_readout_serialization_and_loss(tmp_path):
    rng=np.random.default_rng(28);n=400;ids=np.tile(np.arange(40),10);times=np.repeat(np.arange(10),40)
    target=rng.integers(0,2,n);scores=np.c_[.2+.5*target+rng.uniform(0,.15,n),rng.uniform(size=n)];ctx=rng.normal(size=(n,10))
    model,detail=c.fit(scores,ctx,target,times,ids,c.SETTINGS);path=tmp_path/'model.joblib';model.save(path)
    prediction=model.predict(scores,ctx);np.testing.assert_array_equal(prediction,c.ContextPairwiseModel.load(path).predict(scores,ctx))
    np.testing.assert_array_equal(prediction[:20],np.array([model.predict_one(*s,h) for s,h in zip(scores[:20],ctx[:20])],dtype=np.float32))
    assert detail['final_pair_loss']<=detail['initial_pair_loss'] and detail['new_base_fits']==0


def test_no_extra_or_nonfinite_context():
    with pytest.raises(ValueError):c.features(np.ones((4,2)),np.ones((4,11)))
    with pytest.raises(ValueError):c.features(np.ones((4,2)),np.full((4,10),np.nan))
