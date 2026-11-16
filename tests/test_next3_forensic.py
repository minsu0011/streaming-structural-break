import numpy as np
import pytest
from src.next3.forensic import positive_pair_credit
from src.scoring.ts_auc import ts_auc


def test_pair_attribution_matches_explicit_tied_pair_comparisons():
    y = np.array([1,0,1,0,1,0,1,0,1])
    p = np.array([.5,.5,.7,.1,.2,.7,.1,.5,.5])
    t = np.array([0,0,0,0,1,1,1,1,2])
    wins, mass = positive_pair_credit(y,p,t)
    for i in range(len(y)):
        negatives = [j for j in range(len(y)) if t[j] == t[i] and y[j] == 0] if y[i] else []
        assert mass[i] == len(negatives)
        assert wins[i] == sum(float(p[i]>p[j])+.5*float(p[i]==p[j]) for j in negatives)
    assert wins.sum()/mass.sum() == pytest.approx(ts_auc(y,p,t), abs=1e-15)


def test_pair_attribution_permutation_and_group_sum():
    rng = np.random.default_rng(123)
    y = rng.integers(0,2,300)
    p = rng.integers(0,5,300)/4
    t = rng.integers(0,10,300)
    wins, mass = positive_pair_credit(y,p,t)
    order = rng.permutation(len(y))
    other, pairs = positive_pair_credit(y[order],p[order],t[order])
    np.testing.assert_array_equal(other,wins[order])
    np.testing.assert_array_equal(pairs,mass[order])
    groups = np.arange(len(y))%3
    assert sum(wins[groups==g].sum()/mass.sum() for g in range(3)) == pytest.approx(ts_auc(y,p,t), abs=1e-15)
