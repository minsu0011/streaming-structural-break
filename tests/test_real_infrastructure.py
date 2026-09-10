import json
import os
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import pytest
from src.data.audit import raw_manifest
from src.data.grouping import historical_signature,group_histories
from src.features.config import CONFIG
from src.features.cache import cache_version
from src.features.streaming import replay,feature_version
from src.validation.splits import freeze_group_split,FrozenSplit
from src.validation.research import contribution_weights,fit_fold
from src.validation.adaptive import ResearchBudget,Candidate,select_next
from tests.test_research_guards import engineering_folds

def test_manifest_mtime_is_not_identity(tmp_path):
    file=tmp_path/'X_train.parquet';pd.DataFrame({'value':[1.,2.]}).to_parquet(file)
    manifest=tmp_path/'manifest.json';before=raw_manifest(tmp_path,manifest)
    os.utime(file,ns=(file.stat().st_atime_ns,file.stat().st_mtime_ns+10_000_000))
    after=raw_manifest(tmp_path,manifest)
    assert before['files'][0]['sha256']==after['files'][0]['sha256']
    pd.DataFrame({'value':[3.,4.]}).to_parquet(file)
    with pytest.raises(RuntimeError,match='Immutable'):raw_manifest(tmp_path,manifest)

def test_affine_and_near_clones_share_frozen_allocation(tmp_path):
    rng=np.random.default_rng(5);h=rng.normal(size=256)
    histories=[h,3*h+7,h+rng.normal(scale=.001,size=256),rng.normal(size=256)]
    rows=[];fingerprints=[]
    for i,history in enumerate(histories):
        signature,fingerprint=historical_signature(history)
        rows.append({'dataset_id':i,**signature});fingerprints.append(fingerprint)
    metadata=pd.DataFrame(rows)
    groups,report=group_histories(metadata,histories,fingerprints,tmp_path/'audit')
    assert groups.group_id.tolist()==[0,0,0,3]
    assert not report['target_or_tau_used'] and not report['all_pairs_matrix_created']
    split=freeze_group_split(groups,tmp_path/'split',group_policy_sha256=report['policy_sha256'],raw_sha256='fixture')
    assert len({split.assignment(i) for i in range(3)})==1
    with pytest.raises(RuntimeError):split.assignment(100)
    changed=groups.copy();changed.loc[2,'group_id']=2
    with pytest.raises(RuntimeError):freeze_group_split(changed,tmp_path/'split',group_policy_sha256=report['policy_sha256'],raw_sha256='fixture')
    with pytest.raises(ValueError):group_histories(metadata.assign(tau=0),histories,fingerprints,tmp_path/'bad')

def test_feature_config_changes_hash_and_causal_output():
    variant=CONFIG.variant(scales=(3,12,48),lags=(1,3,6,12))
    assert CONFIG.digest()!=variant.digest()
    assert feature_version(CONFIG)!=feature_version(variant)
    assert cache_version(CONFIG)!=cache_version(variant)
    h=np.arange(100,dtype=float)%7;o=np.linspace(-2,2,40)
    a=np.array(list(replay(h,o,config=variant)))
    b=np.array(list(replay(h,np.r_[o[:20],np.ones(100)*1000],config=variant)))
    np.testing.assert_array_equal(a[:20],b[:20])
    assert not np.array_equal(a,np.array(list(replay(h,o))))

def test_pair_weighting_and_validation_labels_do_not_fit():
    y=np.array([0,0,0,1,0,1]);t=np.array([0,0,0,0,1,1])
    weights=contribution_weights(y,t)
    assert weights[3]==3*weights[0] and weights[4]==weights[5]
    assert np.isclose(weights.mean(),1)
    with pytest.raises(ValueError):contribution_weights([0,1],[0,1])
    folds=engineering_folds()
    for k,frame in enumerate(folds):frame['target']=(frame.time_online+k)%2
    first,pred,_=fit_fold(folds,1,'M1_logistic','ABCD','S3')
    folds[1]['target']=1-folds[1].target
    second,pred2,_=fit_fold(folds,1,'M1_logistic','ABCD','S3')
    np.testing.assert_array_equal(first.linear_weight,second.linear_weight)
    np.testing.assert_array_equal(pred,pred2)

def test_wallclock_budget_survives_resume_and_reserves_buffer(tmp_path):
    budget=ResearchBudget(tmp_path/'budget.json',hours=10,start='2026-09-08T06:49:52+00:00')
    assert budget.can_start(60,datetime(2026,9,8,16,0,tzinfo=timezone.utc))
    assert not budget.can_start(0,datetime(2026,9,8,16,5,tzinfo=timezone.utc))
    resumed=ResearchBudget(tmp_path/'budget.json',hours=100)
    assert resumed.state['deadline_utc']==budget.state['deadline_utc']
    attempts={Candidate(model).identity for model in ('M1_logistic','M2_ridge','M3_histgb','M4_lightgbm','M5_xgboost')}
    result={'candidate':Candidate('M4_lightgbm').__dict__,'model':'M4_lightgbm','status':'COMPLETE','median':.7,'std':.01,'worst_fold':.68}
    next_candidate=select_next([result],attempts)
    assert next_candidate.phase=='sampling_weighting' and next_candidate.identity not in attempts
