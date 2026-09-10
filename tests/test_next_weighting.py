import numpy as np
from src.next.fitting import weights_from_training
from src.validation.research import contribution_weights


def test_weighting_uses_exact_train_pair_contribution():
    ids = np.repeat(np.arange(6), 4)
    times = np.tile(np.arange(4), 6)
    y = np.array([int(t >= sid % 4 and sid % 3 != 0) for sid,t in zip(ids,times)])
    actual = weights_from_training(y, times, ids, 'W3')
    np.testing.assert_array_equal(actual, contribution_weights(y,times))
    for policy in ['W2','W3']:
        w = weights_from_training(y,times,ids,policy)
        totals = np.array([w[times == t].sum() for t in range(4)])
        pairs = np.array([sum(y[times==t])*sum(1-y[times==t]) for t in range(4)])
        np.testing.assert_allclose(totals/totals.sum(), pairs/pairs.sum(), rtol=0, atol=1e-15)
    for t in range(4):
        if np.any(y[times==t]) and np.any(1-y[times==t]):
            np.testing.assert_allclose(actual[(times==t)&(y==0)].sum(), actual[(times==t)&(y==1)].sum(), atol=1e-15)


def test_inverse_series_density_equalizes_train_series_totals():
    ids = np.repeat(np.arange(3), [2,4,8])
    times = np.concatenate([np.arange(n) for n in [2,4,8]])
    y = times % 2
    weight = weights_from_training(y,times,ids,'W1')
    np.testing.assert_allclose([weight[ids==i].sum() for i in range(3)], [len(ids)/3]*3)
