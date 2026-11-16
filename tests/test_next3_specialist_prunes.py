import numpy as np
import pytest
from src.next3.artifacts import BASE,read_json
from src.next3.frozen import candidate,C205
from src.next2.io import validate
from src.next.extensions import ExtendedFeatureState


@pytest.mark.parametrize('kind',['P1_LAGS_1_2','P2_LAGS_4_8','P3_LOCAL_ONLY','P4_BAYES_ONLY'])
def test_pruned_live_inputs_are_exact_nonempty_frozen_subsets(kind):
    original,_=candidate(C205)
    selected=read_json(BASE/'hypotheses'/f'{kind}.json')['candidate']
    validate(selected)
    names=selected['keep_names']
    assert len(names) in (8,12,16) and len(set(names))==len(names)
    indices=[original['keep_names'].index(name) for name in names]
    rng=np.random.default_rng(203)
    h=rng.normal(size=256).astype(np.float32);o=rng.normal(size=32).astype(np.float32)
    full=ExtendedFeatureState(h,original['keep_names'],original['extension'])
    smaller=ExtendedFeatureState(h,names,selected['extension'])
    for point in o:
        np.testing.assert_array_equal(smaller.update(point),full.update(point)[indices])
