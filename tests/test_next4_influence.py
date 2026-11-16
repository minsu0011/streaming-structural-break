import numpy as np
from src.next4.influence import time_credits,delete_one_scores
from src.scoring.ts_auc import ts_auc


def test_time_pair_mass_credits_sum_to_exact_score_change():
    rng=np.random.default_rng(44);ids=np.repeat(np.arange(7),500);t=np.tile(np.arange(500),7)
    y=((ids%2==0)&(t>10+ids)).astype(np.uint8)
    p=rng.random(len(y)).astype(np.float32);ref=rng.random(len(y)).astype(np.float32)
    rows=time_credits(y,p,ref,t)
    assert len(rows)==7
    np.testing.assert_allclose(sum(r['fraction_of_fold_pair_mass'] for r in rows),1.,rtol=0,atol=1e-14)
    np.testing.assert_allclose(sum(r['contribution_to_fold_delta'] for r in rows),ts_auc(y,p,t)-ts_auc(y,ref,t),rtol=0,atol=1e-14)
    # An unweighted mean of time-bin deltas is not the official pair-mass metric.
    assert abs(np.mean([r['bin_delta'] for r in rows if r['bin_delta'] is not None])-(ts_auc(y,p,t)-ts_auc(y,ref,t)))>1e-5


def test_exact_group_deletions_match_independent_row_removal_with_ties():
    ids=np.repeat([19,29,39,49,59],8);t=np.tile(np.arange(8),5)
    y=((ids%3==1)&(t>=2)).astype(np.uint8);p=((np.arange(40)*7)%5).astype(np.float32)/5
    groups,scores=delete_one_scores(y,p,t,ids)
    for group,score in zip(groups,scores[1:]):
        keep=ids!=group
        assert abs(score-ts_auc(y[keep],p[keep],t[keep]))<1e-14
