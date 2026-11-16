"""Independent direct-window statistics and causal state checks."""
import numpy as np
import pytest
from src.next5 import windowed as w,innovation


@pytest.mark.parametrize('kind',w.KINDS)
def test_hundred_suffix_length_and_point_replay_cases(kind):
    rng=np.random.default_rng(20260908);settings={'version':'next5_windowed_v1','kind':kind}
    for case in range(100):
        h=rng.standard_t(5,192).astype(np.float32);o=rng.normal(size=80).astype(np.float32);cutoff=case%70+1
        changed=o.copy();changed[cutoff:]=rng.normal(0,20,len(o)-cutoff)
        full=w.make_state(h,settings).replay(o);other=w.make_state(h,settings).replay(changed)
        short=w.make_state(h,settings).replay(o[:cutoff]);state=w.make_state(h,settings)
        pointwise=np.asarray([state.update(v).copy() for v in o])
        np.testing.assert_array_equal(full[:cutoff],other[:cutoff]);np.testing.assert_array_equal(full[:cutoff],short)
        np.testing.assert_array_equal(full,pointwise);assert np.isfinite(full).all()


@pytest.mark.parametrize('kind',w.KINDS)
def test_direct_window_oracle_after_multiple_ring_wraps(kind):
    rng=np.random.default_rng(20260908);h=rng.standard_t(5,192).astype(np.float32);o=rng.normal(size=400).astype(np.float32)
    state=w.make_state(h,{'version':'next5_windowed_v1','kind':kind});reference=state.args[1];args,_=innovation.prepare(h)
    energy=np.asarray([innovation.step(v,*args).copy()**2 for v in o]);actual=state.replay(o)
    for index in (0,15,16,31,32,63,64,127,128,129,255,399):
        expected=[]
        for channel in range(2):
            ratios=[]
            for window in w.WINDOWS:
                recent=energy[max(0,index-window+1):index+1,channel]
                ratio=np.log(max(recent.mean(),1e-8)/reference[channel,0]);ratios.append(ratio)
                if kind=='scale_ratio':expected.extend([ratio,max(-ratio,0.),max(ratio,0.)])
                elif kind=='trimmed_energy':expected.extend([np.log(max(np.minimum(recent,reference[channel,1]).mean(),1e-8)/reference[channel,2]),np.log(recent+1e-4).mean()-reference[channel,3]])
            if kind=='disagreement':
                for fast,slow in ((0,3),(1,2)):
                    difference=ratios[fast]-ratios[slow]
                    expected.extend([difference,abs(difference),min(max(ratios[fast],0.),max(ratios[slow],0.)),min(max(-ratios[fast],0.),max(-ratios[slow],0.))])
        np.testing.assert_allclose(actual[index],expected,rtol=1e-6,atol=1e-6)


@pytest.mark.parametrize('kind',w.KINDS)
def test_constant_H_and_large_point_do_not_make_infinite_log_ratio(kind):
    state=w.make_state(np.ones(192,dtype=np.float32),{'version':'next5_windowed_v1','kind':kind})
    assert np.isfinite(state.replay(np.r_[np.zeros(128),1e8,np.ones(128)].astype(np.float32))).all()
