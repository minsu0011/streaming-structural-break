"""Independent frozen batch oracle versus resident scalar execution adapter."""
from dataclasses import replace
import numpy as np
import pytest
from src.next5.artifacts import BASE,read_json
from src.next5.synthetic import load_reference,load_candidate,predict_model,reference_predictions,combine_candidate
from src.next5.runtime import SharedResearchBaseline,PreparedComposite,feature_step
from src.next5.model import feature_module

CANDIDATES=sorted(p.stem for p in (BASE/'registrations').glob('*.json'))


@pytest.fixture(scope='module')
def references():return {scope:load_reference(scope,0)[0] for scope in ('full','screen')}


@pytest.mark.parametrize('cid',CANDIDATES)
def test_frozen_feature_coordinates_and_float64_composite_match_scalar_adapter(cid,references):
    scope='full' if (BASE/'experiments'/cid/'full/RESULTS.json').exists() else 'screen'
    expert,registration,_=load_candidate(cid,scope,0);refs=references[scope]
    prepared=PreparedComposite(refs['J102'],refs['C205'],expert,registration['combination']);rng=np.random.default_rng(20260908)
    for case in range(3):
        historical=(rng.standard_t(5,192+case*64)*(.5+case)).astype(np.float32)
        online=rng.normal(size=96).astype(np.float32);online[30:]*=.5 if case==0 else 2.;online[50]+=8.
        module=feature_module(expert.module_name);expected_features=module.make_state(historical,expert.settings).replay(online)
        args=module.make_state(historical,expert.settings).args;update=feature_step(expert.module_name,expert.settings)
        actual_features=np.asarray([update(float(value),*args).copy() for value in online])
        np.testing.assert_array_equal(actual_features,expected_features)
        old=reference_predictions(refs,historical,online);prediction=predict_model(expert,historical,online)
        expected=combine_candidate(prediction,old['BL80'],old['C205'],registration)
        state=prepared.new_state(historical);before=state.state_array_bytes
        actual=np.asarray([state.predict_one(value) for value in online])
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-15)
        assert state.state_array_bytes==before
        short=prepared.new_state(historical);cutoff=37
        np.testing.assert_array_equal(actual[:cutoff],np.asarray([short.predict_one(value) for value in online[:cutoff]]))


@pytest.mark.parametrize('scope',['full','screen'])
def test_shared_baseline_retains_float64_and_matches_exact_unshared_reference(scope,references):
    refs=references[scope];rng=np.random.default_rng(20260908);h=rng.normal(size=512).astype(np.float32);o=rng.normal(size=300).astype(np.float32)
    expected=reference_predictions(refs,h,o)['BL80'];state=SharedResearchBaseline(refs['J102'],refs['C205']).new_state(h)
    actual=np.array([state.predict_one(v) for v in o]);np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-15)
    assert actual.dtype==np.float64


def test_unmatched_training_membership_rejected_before_inference(references):
    refs=references['full'];bad=replace(refs['C205'],training={**refs['C205'].training,'training_ids_sha256':'wrong'})
    with pytest.raises(ValueError):SharedResearchBaseline(refs['J102'],bad)


def test_invalid_current_observation_does_not_advance_recurrences(references):
    refs=references['full'];prepared=SharedResearchBaseline(refs['J102'],refs['C205']);h=np.sin(np.arange(256)*.2).astype(np.float32)
    state=prepared.new_state(h)
    for bad in (np.nan,np.inf,-np.inf,1e100):
        with pytest.raises(ValueError):state.predict_one(bad)
    assert state.predict_one(.5)==prepared.new_state(h).predict_one(.5)
