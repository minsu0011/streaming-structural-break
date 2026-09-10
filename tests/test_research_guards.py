import json
import numpy as np
import pandas as pd
import pytest
from src.features.cache import build_cache,read_fold
from src.features.streaming import FEATURE_NAMES,StreamingFeatureState,replay
from src.validation.splits import assignment
from src.validation.research import fit_fold,summarize,sample_frame
from src.validation.diagnostics import profile
from scripts.run_research import admission
from tests.test_loader import make_data
from src.models.baselines import BaselineDetector,BASELINES

def test_admission_blocks_missing_evidence(tmp_path):
    with pytest.raises(RuntimeError): admission(tmp_path)

def test_cache_excludes_seal_and_detects_tampering(tmp_path):
    raw=tmp_path/'raw';raw.mkdir(); make_data(raw)
    directory=build_cache(raw,tmp_path/'cache')
    manifest=json.loads((directory/'MANIFEST.json').read_text())
    assert manifest['status']=='COMPLETE' and not manifest['seal_extracted']
    expected_ids={sid for sid in range(4) if assignment(sid)[0]=='DEVELOPMENT_POOL'}
    frames=[pd.read_parquet(directory/s['name']) for s in manifest['shards']]
    merged=pd.concat(frames)
    assert set(merged.dataset_id)==expected_ids
    assert all(merged[name].dtype==np.float32 for name in FEATURE_NAMES)
    assert build_cache(raw,tmp_path/'cache')==directory
    for sid,part in merged.groupby('dataset_id'):
        historical=np.arange(sid*7,sid*7+3,dtype=np.float32)
        online=np.arange(sid*7+3,sid*7+7,dtype=np.float32)
        np.testing.assert_array_equal(part.loc[:,FEATURE_NAMES].to_numpy(),np.array(list(replay(historical,online))))
        for name in BASELINES:
            detector=BaselineDetector(name,historical)
            expected=np.array([detector.predict_one(x) for x in online],dtype=np.float32)
            np.testing.assert_array_equal(part[name].to_numpy(),expected)
    shard=directory/manifest['shards'][0]['name']
    shard.write_bytes(shard.read_bytes()+b'corruption')
    with pytest.raises(RuntimeError):build_cache(raw,tmp_path/'cache')

def engineering_folds():
    rng=np.random.default_rng(2);folds=[]
    for fold in range(5):
        sid=next(i for i in range(1000) if assignment(i)==('DEVELOPMENT_POOL',fold))
        frame=pd.DataFrame(rng.normal(size=(20,len(FEATURE_NAMES))).astype(np.float32),columns=FEATURE_NAMES)
        frame['dataset_id']=sid;frame['time_online']=np.arange(20);frame['target']=np.arange(20)%2
        frame['sample_S1']=1;frame['sample_S2']=1
        folds.append(frame)
    return folds

def test_fold_exclusion_and_target_invariance():
    folds=engineering_folds()
    model,pred,_=fit_fold(folds,2,'M1_logistic','ABCD','S2')
    assert pred.dtype==np.float32  # Match official wire precision when scoring CV.
    changed=[f.copy() for f in folds];changed[2]['target']=1-changed[2].target
    second,pred2,_=fit_fold(changed,2,'M1_logistic','ABCD','S2')
    np.testing.assert_array_equal(model.linear_weight,second.linear_weight)
    np.testing.assert_array_equal(pred,pred2)
    changed[2]['dataset_id']=changed[0].dataset_id.iloc[0]
    with pytest.raises(RuntimeError):fit_fold(changed,2,'M1_logistic','ABCD','S2')
    with pytest.raises(ValueError):summarize([.5,.6])
    with pytest.raises(ValueError):sample_frame(folds[0],'unknown')

def test_no_break_diagnostic_does_not_claim_auc():
    metadata=pd.DataFrame({'dataset_id':[1,2],'tau_index':[np.nan,2],'tau_fraction':[np.nan,.5],
        'has_break':[False,True],'historical_length':[1000,3000],'online_length':[4,4]})
    rows=profile([0,0,0,0,0,0,1,1],[.1,.2,.1,.3,.1,.1,.8,.9],[0,1,2,3]*2,[1]*4+[2]*4,metadata)
    result=next(r for r in rows if r['slice']=='no_break')
    assert result['ts_auc'] is None and result['negative']==4
