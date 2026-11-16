import src.next9
import numpy as np
from src.next9.sampling_phase import phase_counts,eligible_group_frequencies,OFFSETS


def test_short_tail_cohort_excluded_and_family_counts_conserved():
    t=np.r_[np.arange(16),np.arange(400,403)]
    s=np.zeros((len(t),2,4),dtype=np.int8)
    s[:,0,0]=t%3;s[:,1,3]=4
    counts,totals=phase_counts(s,t)
    freq,eligible=eligible_group_frequencies(counts,totals)
    assert eligible[0] and eligible[1] and not eligible[2] and not eligible[7]
    assert totals[7,8]==3 and not freq[7].any()
    for offset,width in zip(OFFSETS,[4,10,7,5]):
        np.testing.assert_array_equal(counts[:,:,:,offset:offset+width].sum(axis=3),
                                      np.broadcast_to(totals[:,:,None],(8,9,2)))
        np.testing.assert_allclose(freq[eligible,:,:,offset:offset+width].sum(axis=3),1)


def test_phase_locked_short_reversal_visible_in_full_stream():
    t=np.arange(80);s=np.zeros((80,1,4),dtype=np.int8)
    s[:,0,3]=np.where(t%8==1,2,0)
    counts,totals=phase_counts(s,t);freq,eligible=eligible_group_frequencies(counts,totals)
    assert freq[0,0,0,23]==0 and freq[0,1,0,23]==1
    assert freq[0,8,0,23]==.125
