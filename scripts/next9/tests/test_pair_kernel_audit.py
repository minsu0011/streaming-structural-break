import src.next9
import numpy as np
import pytest
from src.next9.pair_kernel_audit import prepare_pairs,weighted_pairs


def literal(y,p,t,g,weights):
    result=[]
    for w in weights:
        numerator=0.;denominator=0
        for a in range(len(y)):
            for b in range(len(y)):
                if t[a]==t[b] and y[a]==1 and y[b]==0:
                    pair=int(w[g[a]])*int(w[g[b]])
                    denominator+=pair
                    numerator+=pair*(int(p[a]>p[b])+.5*int(p[a]==p[b]))
        result.append(numerator/denominator if denominator else np.nan)
    return np.asarray(result)


def test_literal_ties_disconnected_times_and_zero_draws():
    y=np.array([1,0,1,0,1,0,1,0],dtype=np.uint8)
    p=np.array([.5,.5,.9,.1,.2,.8,.9,.9])
    t=np.array([0,0,1,1,1,1,7,9]);g=np.array([0,1,0,1,2,3,2,3])
    w=np.array([[1,1,1,1],[2,0,0,2],[0,0,0,0],[0,4,0,0],[1,3,0,0]])
    unique,count,win2=prepare_pairs(y,p,t,g)
    actual,pairs=weighted_pairs(count,win2,w)
    np.testing.assert_array_equal(actual,literal(y,p,t,g,w))
    assert actual[0]==.7  # 0.5 + four-pair time block wins 3 of 4, over five pairs.
    assert pairs[0]==5


def test_duplicate_group_time_rejected():
    with pytest.raises(ValueError,match='one observation'):
        prepare_pairs([1,0],[.8,.2],[0,0],[1,1])


def test_integer_range_guard_and_empty():
    _,c,k=prepare_pairs([],[],[],[])
    actual,pairs=weighted_pairs(c,k,np.empty((3,0)))
    assert np.isnan(actual).all() and not pairs.any()
    with pytest.raises(OverflowError):
        weighted_pairs(np.ones((1,1),dtype=np.int64),np.ones((1,1),dtype=np.int64),[[100000000]])
