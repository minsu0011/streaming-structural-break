import numpy as np
import pytest
from src.features.config import CONFIG
from src.models.representations import calibration_policy,make_state


@pytest.mark.parametrize('prefix',['ar_residual_input_','ar_order8_'])
def test_wider_output_clip_changes_only_predeclared_saturation(prefix):
    config=CONFIG.variant(normalization='median_mad',scales=(5,20,160));representation=prefix+'historical_median_mad'
    old=calibration_policy(representation);new=calibration_policy(representation+'_clip24')
    assert old.clip==12 and new.clip==24 and {k:v for k,v in vars(old).items() if k!='clip'}=={k:v for k,v in vars(new).items() if k!='clip'}
    rng=np.random.default_rng(461);h=rng.standard_t(4,size=600).astype(np.float32)
    before=make_state(h,config,representation);after=make_state(h,config,representation+'_clip24')
    for point in np.r_[rng.normal(size=30),np.full(200,100.)]:
        left=before.update_and_get(point).copy();right=after.update_and_get(point).copy()
        mask=before.inner.mask
        np.testing.assert_array_equal(left[~mask],right[~mask]);np.testing.assert_array_equal(left[mask],np.clip(right[mask],-12,12))
        assert np.isfinite(right).all() and np.all(abs(right[mask])<=24)
