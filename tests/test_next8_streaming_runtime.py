"""Real-engine scalar parity, model coordinates and complete saved memory."""
import src.next8
import copy
import numpy as np
import pytest
import joblib
from src.next8.features import FeatureState,N_FEATURES
from src.next8.synthetic_controls import generate_block,SyntheticCache,matched_path
from src.next8.runtime_features import raw_arguments,normalized_step
from src.next8.runtime_head import conditions
from src.next8.streaming_runtime import PreparedJoint,JointState
from src.next8.joint_models import joint_design,prune_training_design,model_design
from src.next8.learning import fit_logistic,fit_tree
from src.next8.data import FitScope
from src.next8.graph_states import predict_graph_state
from src.next8.transitions import FactorizedTransition
from src.next8.common import ids_hash


@pytest.fixture(scope='module')
def actual_cache(tmp_path_factory):
    p=tmp_path_factory.mktemp('next8_runtime_actual_engine')
    generate_block(0,p)
    c=SyntheticCache(p)
    h,online=matched_path('Gaussian',0,512)
    return c,h,online


@pytest.mark.parametrize('used',[(0,6,15),(51,64),(77,81),(93,105),(117,150),(0,51,77,93,117)])
def test_pruned_raw_kernel_matches_frozen_feature_engine(used):
    rng=np.random.default_rng(882);h=rng.normal(size=512).astype(np.float32);online=rng.normal(size=65).astype(np.float32)
    reference=FeatureState(h);initial=FeatureState.restore(reference)
    expected,_=reference.replay(online)
    engine,norm=raw_arguments(initial,used)
    actual=np.asarray([normalized_step(float(x),engine,norm)[list(used)].copy() for x in online])
    np.testing.assert_array_equal(actual,expected[:,used])


def selection_fixture(scope):
    return {'scope':scope.provenance(),'selection_ids':sorted(scope.training_ids),
        'stable_features':[6,93],'variable_features':[0],
        'classifications':[{'feature':6,'classification':'INVARIANT_DIRECTION'},
            {'feature':93,'classification':'STABLE_NONLINEAR'},
            {'feature':0,'classification':'CONDITIONAL_VARIABLE_CANDIDATE'}],
        'orientation_state_specs':{'0':{'kind':'OWN_F'}}}


def confident_graph(scope):
    p=np.array([.05,.05,.90])
    return FactorizedTransition(p,np.tile(p,(3,1)),np.full(3,128),
        [np.tile(p,(3,3,1))],[np.full((3,3),128)],(105,),{**scope.provenance(),'fixture_only':True})


def fitted_fixture(cache,method,*,state_kind='OWN_F',unknown_H=False,early=False):
    ids=cache.series['dataset_id'];scope=FitScope(frozenset(map(int,ids)),frozenset({-1}),frozenset({-1}),'runtime_fixture')
    selection=selection_fixture(scope);rows=np.arange(len(cache.values));graph=confident_graph(scope)
    if state_kind!='OWN_F':
        index=2 if state_kind=='H_LAG1' else 4
        selection['orientation_state_specs']['0']={'kind':state_kind,'context_index':index,
            'low':.01,'high':.02,'minimum':2. if unknown_H else -1.,'maximum':3. if unknown_H else 1.,
            'valid':True,'training':scope.provenance()}
    cross=np.full((len(rows),12),3,dtype=np.int8);cross[:,0]=predict_graph_state(cache,rows,0,graph)[0]
    # Artificial supervision tests execution arithmetic only, never enters RP.
    rng=np.random.default_rng(819);y=(rng.random(len(rows))<.5+.1*np.tanh(cache.values[:,0])).astype(np.uint8)
    design=joint_design(cache,rows,selection,method,cross)
    if early:
        from src.next8.learning import Design
        keep=cache.meta['time_online'][rows]<=199
        original_training=Design(design.values[keep],design.names,design.ids[keep]);labels=y[keep]
    else:original_training=design;labels=y
    training,cols,_=prune_training_design(original_training)
    model=fit_tree(training,labels,scope) if method=='M3' else fit_logistic(training,labels,scope)
    model.training.update({'original_input_names':list(design.names),'retained_input_indices':cols,'early_training_cutoff':199 if early else None})
    return model,selection,graph,cross


@pytest.mark.parametrize('method',['M1','M2','M3','M4','M5'])
def test_stream_offline_model_and_state_parity(actual_cache,method):
    cache,h,online=actual_cache;model,selection,graph,cross=fitted_fixture(cache,method)
    rows=np.arange(1024);design=joint_design(cache,rows,selection,method,cross[rows])
    expected_design=model_design(model,design)
    native=model.predict_proba(expected_design)[:,1]
    expected=native.astype(np.float32)
    prepared=PreparedJoint(model,selection,method,{0:graph},resident=False)
    state=prepared.new_state(h);actual=[]
    for k,x in enumerate(online):
        actual.append(state.predict_one(x))
        np.testing.assert_array_equal(state.last_design,expected_design.values[k])
        assert abs(state.unrounded_probability-native[k])<1e-12
        if method in ('M3','M5'):
            assert state.forecast_graph_condition[0]==cross[k,0]
    np.testing.assert_allclose(actual,expected,rtol=0,atol=np.finfo(np.float32).eps)
    assert state.counter==1024


def test_state_save_restore_and_no_cross_series_state(actual_cache,tmp_path):
    cache,h,online=actual_cache;model,selection,graph,cross=fitted_fixture(cache,'M5')
    prepared=PreparedJoint(model,selection,'M5',{0:graph},resident=True)
    first=prepared.new_state(h);first.replay(online[:73]);saved=first.snapshot()
    joblib.dump(saved,tmp_path/'state.joblib');restored=joblib.load(tmp_path/'state.joblib')
    a=first.replay(online[73:]);b=restored.replay(online[73:]);np.testing.assert_array_equal(a,b)
    independent=prepared.new_state(h);fresh=prepared.new_state(h)
    independent.replay(-online[:10])
    np.testing.assert_array_equal(fresh.replay(online),prepared.new_state(h).replay(online))
    assert saved.counter==73


def test_future_suffix_and_chunk_boundaries_do_not_change_prefix(actual_cache):
    cache,h,online=actual_cache;model,selection,graph,cross=fitted_fixture(cache,'M4')
    prepared=PreparedJoint(model,selection,'M4',{0:graph},resident=False)
    full=prepared.new_state(h).replay(online)
    state=prepared.new_state(h);chunk=np.r_[state.replay(online[:37]),state.replay(online[37:200]),state.replay(online[200:])]
    np.testing.assert_array_equal(chunk,full)
    modified=online.copy();modified[200:]*=-12
    np.testing.assert_array_equal(prepared.new_state(h).replay(modified)[:200],full[:200])


def test_short_H_zero_feature_and_unknown_policy(actual_cache):
    cache,_,online=actual_cache;model,selection,graph,cross=fitted_fixture(cache,'M5')
    prepared=PreparedJoint(model,selection,'M5',{0:graph},resident=False)
    state=prepared.new_state(np.arange(12,dtype=np.float32))
    prediction=state.replay(online[:16])
    assert state.short_H and np.isfinite(prediction).all()
    assert (state.current_feature==0).all() and (state.own_orientation_condition==3).all() and (state.forecast_graph_condition==3).all()
    before=state.counter
    with pytest.raises(ValueError):state.predict_one(np.nan)
    assert state.counter==before


@pytest.mark.parametrize('method,kind,unknown',[('M4','H_TAIL3',False),('M4','H_LAG1',True),('M2','H_LAG1',True)])
def test_H_condition_and_UNKNOWN_gating_match_offline(actual_cache,method,kind,unknown):
    cache,h,online=actual_cache
    model,selection,graph,cross=fitted_fixture(cache,method,state_kind=kind,unknown_H=unknown)
    rows=np.arange(256);design=model_design(model,joint_design(cache,rows,selection,method,cross[rows]))
    state=PreparedJoint(model,selection,method,{0:graph},resident=False).new_state(h)
    observed=[]
    for k,x in enumerate(online[:256]):
        observed.append(state.predict_one(x));np.testing.assert_array_equal(state.last_design,design.values[k])
    np.testing.assert_allclose(observed,model.predict_proba(design)[:,1].astype(np.float32),rtol=0,atol=np.finfo(np.float32).eps)


def test_integrated_baseline_state_survives_restore(actual_cache,tmp_path):
    from types import SimpleNamespace
    from src.next8.early_reference import fit_early_reference
    from src.next8.common import ROOT,read_json
    from src.next5.model import feature_module
    from src.next5.runtime import SharedResearchBaseline
    from src.next8.streaming_runtime import IntegratedState
    cache,h,online=actual_cache;model,selection,graph,cross=fitted_fixture(cache,'M5',early=True)
    scope=FitScope(frozenset(selection['selection_ids']),frozenset({-1}),frozenset({-1}),'runtime_fixture')
    class Guard:
        split=SimpleNamespace(digest='SYNTHETIC_EXECUTION_FIXTURE')
        def partition(self,a,b):assert set(a).isdisjoint(b)
    rng=np.random.default_rng(8882);metadata={**cache.meta,'target':rng.integers(0,2,len(cache.values),dtype=np.uint8)}
    references=[]
    for kind in ('J102','C205'):
        reg=read_json(ROOT/f'artifacts/next5/references/registrations/REF_{kind}.json')
        width=len(feature_module(reg['feature_module']).feature_names(reg['feature_settings']))
        x=rng.normal(size=(len(cache.values),width)).astype(np.float32)
        references.append(fit_early_reference(x,metadata,scope,reg,Guard()))
    baseline=SharedResearchBaseline(*references);joint=PreparedJoint(model,selection,'M5',{0:graph},resident=True)
    state=IntegratedState(baseline,joint,h,True);state.replay(online[:23])
    saved=state.snapshot();joblib.dump(saved,tmp_path/'integrated.joblib');restored=joblib.load(tmp_path/'integrated.joblib')
    np.testing.assert_array_equal(state.replay(online[23:65]),restored.replay(online[23:65]))
    a=IntegratedState(baseline,joint,h,False).replay(online[:65])
    b=np.asarray([baseline.new_state(h).predict_one(online[0])])
    assert a[0]==b[0]
    bad=copy.deepcopy(joint);bad.selection['scope']['training_ids_sha256']='different'
    bad.selection['selection_ids']=[999]
    with pytest.raises(ValueError,match='membership'):IntegratedState(baseline,bad,h,True)
