"""Fixed cross-feature banks, bounded products and current-prefix independence."""
import copy
import numpy as np
import pytest
from src.next5 import cross
from src.next5.model import feature_module


@pytest.mark.parametrize('kind',cross.KINDS)
def test_hundred_prefix_length_and_pointwise_cases(kind):
    rng=np.random.default_rng(20260908);settings=cross.locked_settings()[kind]
    for case in range(100):
        h=rng.standard_t(5,192).astype(np.float32);o=rng.normal(size=80).astype(np.float32);cutoff=case%70+1
        changed=o.copy();changed[cutoff:]=rng.normal(0,20,len(o)-cutoff)
        full=cross.make_state(h,settings).replay(o);other=cross.make_state(h,settings).replay(changed)
        short=cross.make_state(h,settings).replay(o[:cutoff]);state=cross.make_state(h,settings)
        incremental=np.asarray([state.update(v).copy() for v in o])
        np.testing.assert_array_equal(full[:cutoff],other[:cutoff]);np.testing.assert_array_equal(full[:cutoff],short)
        np.testing.assert_array_equal(full,incremental);assert np.isfinite(full).all() and np.max(np.abs(full[:,-8:]))<=1


@pytest.mark.parametrize('kind',cross.KINDS)
def test_parent_coordinates_are_unchanged_and_products_have_independent_oracle(kind):
    rng=np.random.default_rng(20260908);h=rng.normal(size=192).astype(np.float32);o=rng.normal(size=100).astype(np.float32)
    settings=cross.locked_settings()[kind];parents=[];names=[]
    for parent in settings['parents']:
        module=feature_module(parent['feature_module']);parents.append(module.make_state(h,parent['feature_settings']).replay(o));names.append(module.feature_names(parent['feature_settings']))
    products=np.column_stack([np.tanh(parents[0][:,names[0].index(pair['left'])].astype(float)/pair['left_scale'])*
        np.tanh(parents[1][:,names[1].index(pair['right'])].astype(float)/pair['right_scale']) for pair in settings['interactions']])
    expected=np.column_stack([*parents,products]).astype(np.float32)
    np.testing.assert_array_equal(cross.make_state(h,settings).replay(o),expected)


def test_extra_or_target_driven_coordinate_substitution_rejected():
    settings=copy.deepcopy(cross.locked_settings()['scale_persistence']);settings['interactions'][0]['right_scale']=5.
    with pytest.raises(ValueError):cross.make_state(np.ones(192),settings)
