import json
import numpy as np
import pytest
from src.streaming.training import train_frozen_candidate,sample_indices
from src.streaming.inference import infer

class ForbiddenArray:
    def __array__(self,*args,**kwargs):raise AssertionError('Seal values were inspected')
    def __len__(self):raise AssertionError('Seal length was inspected')
    def __iter__(self):raise AssertionError('Seal values were iterated')

def fixture():
    rng=np.random.default_rng(121)
    for sid in range(4):
        if sid==3:yield sid,ForbiddenArray(),ForbiddenArray(),ForbiddenArray();continue
        h=rng.normal(size=300).astype(np.float32);o=rng.normal(size=120).astype(np.float32)
        if sid:o[60:]*=2
        yield sid,h,o,60 if sid else None

@pytest.mark.parametrize('fast_initialization',[False,True])
def test_frozen_training_excludes_seal_and_runs_live(tmp_path,fast_initialization):
    spec={'status':'FROZEN_PRIMARY','development_ids':[0,1,2],'known_training_ids':[0,1,2,3],'use_fused_inference':True,
        'use_fast_historical_init':fast_initialization,
        'candidate':{'model':'M4_lightgbm','families':'ABCD','sampling':'S3','params':{},'feature_changes':{},'representation':'historical_median_mad_startup'}}
    descriptor=train_frozen_candidate(fixture(),tmp_path,spec)
    assert descriptor['development_series_fitted']==3 and descriptor['seal_series_skipped_before_array_or_label_access']==1
    assert descriptor['seal_rows_fitted']==0 and descriptor['training_rows']==360
    _,h,o,_=next(fixture());pred=list(infer(iter([(h,iter(o))]),tmp_path))
    assert pred[0] is None and len(pred)==len(o)+1 and np.isfinite(pred[1:]).all()
    if fast_initialization:
        wrong=dict(descriptor);wrong['initialization_source_hash']='changed'
        descriptor_path=tmp_path/'model.json';descriptor_path.write_text(json.dumps(wrong),encoding='utf-8')
        with pytest.raises(RuntimeError,match='initialization implementation hash'):next(infer(iter([(h,o)]),tmp_path))
        descriptor_path.write_text(json.dumps(descriptor),encoding='utf-8')
    with pytest.raises(RuntimeError):train_frozen_candidate(list(fixture())[:3],tmp_path/'incomplete',spec)
    # The source descriptor's byte hash is checked before unpickling the artifact.
    path=tmp_path/'model.joblib';path.write_bytes(path.read_bytes()+b'corruption')
    with pytest.raises(RuntimeError,match='byte hash'):next(infer(iter([(h,o)]),tmp_path))

def test_sampling_indices_cover_fixed_policy_boundaries():
    assert len(sample_indices(999,'S1'))==256
    selected=sample_indices(999,'S2')
    for lo,hi in [(0,10),(10,25),(25,50),(50,100),(100,200),(200,400),(400,999)]:
        assert int(((selected>=lo)&(selected<hi)).sum())==min(32,hi-lo)
    assert len(np.unique(selected))==len(selected)

@pytest.mark.parametrize('fused,fast',[(False,False),(True,False),(True,True)])
@pytest.mark.parametrize('prefix',['ar_residual_input_','ar_order2_','ar_order8_','ar8_arch1_'])
def test_frozen_ar_training_uses_explicit_native_deployment(tmp_path,fused,fast,prefix):
    representation=prefix+'historical_median_mad'
    spec={'status':'FROZEN_BACKUP' if prefix=='ar_order8_' else 'FROZEN_PRIMARY','development_ids':[0,1,2],'known_training_ids':[0,1,2,3],'use_fused_inference':fused,'use_fast_historical_init':fast,
        'candidate':{'model':'M4_lightgbm','families':'ABCD','sampling':'S3','params':{},
            'feature_changes':{'normalization':'median_mad','scales':[5,20,160]},'representation':representation}}
    descriptor=train_frozen_candidate(fixture(),tmp_path,spec)
    from src.models.representations import bundle_class,deployment_kind
    assert descriptor['kind']==deployment_kind(representation) and descriptor['seal_rows_fitted']==0
    assert descriptor['status']==('FROZEN_DEV_BACKUP_REFIT' if prefix=='ar_order8_' else 'FROZEN_DEV_PRIMARY_REFIT')
    model=bundle_class(representation).load(tmp_path/'model.joblib');_,h,o,_=next(fixture());state=model.make_state(h)
    if prefix in ('ar_order2_','ar_order8_'):assert model.input_order==int(prefix[8])
    expected=np.array([model.predict_one(state.update_and_get(p)) for p in o],dtype=np.float32)
    actual=np.array(list(infer(iter([(h,iter(o))]),tmp_path))[1:],dtype=np.float32)
    np.testing.assert_array_equal(actual,expected)
