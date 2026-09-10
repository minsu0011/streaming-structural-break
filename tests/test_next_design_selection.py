import numpy as np
from src.next.design_selection import select_reference_families
from src.features.streaming import feature_names
from src.features.config import CONFIG


def test_reference_abcd_selection_excludes_efq_but_preserves_new_statistics():
    config=CONFIG.variant(normalization='median_mad',scales=(5,20,160))
    names=['arch8__'+name for name in feature_names(config)]+['bayes_mean_prior025','glr_mean_max']
    x=np.tile(np.arange(len(names),dtype=np.float32),(3,1))
    actual,selected=select_reference_families(x,names)
    np.testing.assert_array_equal(actual[0],np.r_[np.arange(51),57,58])
    assert len(selected)==53
    assert not any(name in selected for name in ['arch8__fast_slow_mean','arch8__evidence_max','arch8__current_missing'])
