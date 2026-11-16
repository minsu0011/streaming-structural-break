import src.next9
import numpy as np
from src.next9.negative_controls_v2 import generate,arch_innovation_variance


def test_dependence_control_retains_finite_matched_ARCH_second_moment():
    stationary=.1/(1-.85)
    for phi in [0.,-.5]:
        intercept=arch_innovation_variance(phi,0.)
        coefficient=phi**2+arch_innovation_variance(phi,1.)-intercept
        assert coefficient<1
        assert abs(intercept/(1-coefficient)-stationary)<1e-12


def test_variance_matched_control_preserves_paired_H_and_prefix():
    h,a,m=generate('ARCH','NONE',9101604,512)
    hh,b,mm=generate('ARCH','DEPENDENCE_SHIFT',9101604,512)
    np.testing.assert_array_equal(h,hh)
    np.testing.assert_array_equal(a[:m['known_synthetic_onset']],b[:m['known_synthetic_onset']])
    assert np.isfinite(b).all()
