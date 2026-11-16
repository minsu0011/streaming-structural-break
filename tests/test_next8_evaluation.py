"""Alias accounting, sparse domains and finite-family paired uncertainty."""
import src.next8
import numpy as np
from src.next8.evaluation import exact_unique_partition,simultaneous_envelope


def test_exact_aliases_preserve_identical_bootstrap_and_zero_pairs():
    groups=np.repeat(np.arange(128),4);t=np.tile(np.arange(4),128);y=(groups%2).astype(np.uint8)
    p=.2+.6*y
    points,samples,receipt=exact_unique_partition(y,{'a':p,'alias':p.astype(np.float32).astype(float),'b':1-p},t,groups,partition='fixture',draws=40)
    # A float32 round trip is not a bitwise float64 alias, even if its ranks tie.
    assert receipt['aliases']['alias']=='alias'
    np.testing.assert_array_equal(samples['a'],samples['alias'])
    points,samples,receipt=exact_unique_partition(y,{'a':p,'alias':p.copy()},t,groups,partition='fixture',draws=40)
    assert receipt['unique_prediction_arrays']==1 and receipt['aliases']['alias']=='a'
    np.testing.assert_array_equal(samples['a'],samples['alias'])
    points,samples,receipt=exact_unique_partition(np.zeros_like(y),{'a':p},t,groups,partition='none',draws=40)
    assert points['a']['TS_AUC'] is None and receipt['status']=='INSUFFICIENT_SUPPORT' and np.isnan(samples['a']).all()


def test_max_absolute_envelope_uses_shared_centered_draws():
    rng=np.random.default_rng(123);noise=rng.normal(0,.01,2000)
    points=[{'M0':{'TS_AUC':.6},'M1':{'TS_AUC':.61},'M2':{'TS_AUC':.58}} for _ in range(5)]
    samples=[{'M0':np.full(2000,.6),'M1':.61+noise,'M2':.58-2*noise} for _ in range(5)]
    r=simultaneous_envelope(points,samples,methods=('M1','M2'))
    assert abs(r['maximum_absolute_centered_bootstrap_radius']-np.quantile(2*np.abs(noise),.95))<1e-12
    assert r['intervals']['M1']['simultaneous_ci_low']<.01
