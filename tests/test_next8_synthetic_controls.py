import src.next8
import numpy as np
from src.next8.synthetic_controls import matched_path,interventions,generate_block,SyntheticCache


def test_H_lengths_share_one_innovation_path_and_are_not_independent():
    for background in ('Gaussian','AR08','ARCH1','Student_t5'):
        short,a=matched_path(background,3,512);long,b=matched_path(background,3,2048)
        np.testing.assert_array_equal(short,long[-512:]);np.testing.assert_array_equal(a,b)
        assert np.isfinite(long).all() and np.isfinite(a).all()


def test_matched_interventions_preserve_the_entire_pre_event_prefix():
    _,online=matched_path('Gaussian',2,512);variants=interventions(online,2)
    for values in variants:np.testing.assert_array_equal(values[:256],online[:256])
    np.testing.assert_allclose(variants[3][256:],online[256:]*np.float32(np.sqrt(2)),rtol=0,atol=0)
    assert variants[1][256]>online[256]
    assert interventions(online,3)[1][256]<online[256]


def test_same_engine_matched_cache_has_no_cross_intervention_memory(tmp_path):
    generate_block(0,tmp_path);cache=SyntheticCache(tmp_path)
    reference=cache.values[:256]
    for k in range(1,5):np.testing.assert_array_equal(cache.values[k*1024:k*1024+256],reference)
    first=np.arange(5)*1024
    np.testing.assert_array_equal(cache.lag(first,[0,6,51],1),cache.series['seed_values'][:,-1,[0,6,51]])
    assert len(np.unique(cache.meta['dependency_group']))==1
    assert (cache.meta['target'][:3*1024]==0).all()
    assert cache.receipt['real_dataset_access'] is False
