import src.next9
import numpy as np
from src.next9.negative_controls import generate


def test_paired_intervention_preserves_H_and_observed_prefix():
    h,a,meta=generate('AR_0.8','NONE',9101604,512)
    hh,b,m=generate('AR_0.8','MEAN_SHIFT',9101604,512)
    np.testing.assert_array_equal(h,hh)
    onset=meta['known_synthetic_onset']
    np.testing.assert_array_equal(a[:onset],b[:onset])
    np.testing.assert_allclose(b[onset:]-a[onset:],1.,atol=3e-7)


def test_single_observation_shock_has_no_fake_latent_suffix_change():
    h,a,meta=generate('ARCH','NONE',9101605,2048)
    hh,b,m=generate('ARCH','SINGLE_SHOCK',9101605,2048)
    onset=meta['known_synthetic_onset']
    np.testing.assert_array_equal(h,hh)
    np.testing.assert_array_equal(a[:onset],b[:onset])
    np.testing.assert_array_equal(a[onset+1:],b[onset+1:])
    assert abs(float(b[onset]-a[onset])-8)<1e-6
