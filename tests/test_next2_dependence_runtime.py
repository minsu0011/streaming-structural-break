import numpy as np
import pytest
from src.next2.dependence_runtime import CompactDependenceState,NAMES
from src.next2.complement_bank import ComplementState,feature_names


@pytest.mark.parametrize('seed',[1,202,7711,491819])
@pytest.mark.parametrize('case',['gaussian','arch','heavy_tail','constant'])
def test_pruned_dependence_all_selected_features_exact(seed,case):
    rng=np.random.default_rng(seed)
    points=rng.standard_t(3,1600) if case=='heavy_tail' else rng.normal(size=1600)
    if case=='arch':
        for t in range(1,len(points)):
            points[t]=points[t]*np.sqrt(.2+.8*points[t-1]**2)
    if case=='constant':
        points[:]=3.7
    h,o=points[:1024].astype(np.float32),points[1024:].astype(np.float32)
    settings={'family':'dependence','order':8,'cdf_bins':8,'cap':4}
    original=ComplementState(h,settings)
    compact=CompactDependenceState(h)
    full=['next2_complement__'+n for n in feature_names(settings)]
    columns=[full.index(n) for n in NAMES]
    for point in o:
        np.testing.assert_array_equal(compact.update(point),original.update(point)[columns])
    assert len(compact.args[-1][2][0])==2
    assert compact.args[-2].size==0
