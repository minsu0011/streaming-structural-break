"""No-pair directions and group rather than row loss evidence."""
import src.next8
import numpy as np
from src.next8.diagnostics import safe_rank,loss_evidence,equal_fold_loss_summary


def test_zero_pair_direction_is_missing_not_half_or_invariant():
    r=safe_rank(np.zeros(10),np.arange(10),np.zeros(10),np.arange(10))
    assert r['TS_AUC'] is None and r['ci_low'] is None and r['status']=='INSUFFICIENT_SUPPORT'


def test_loss_gain_weights_groups_not_unequal_row_counts():
    groups=np.r_[np.zeros(100,dtype=int),np.arange(1,64)]
    a=np.r_[np.ones(100),np.zeros(63)];b=np.zeros(len(groups))
    result,keys,gain=loss_evidence(groups,a,b)
    assert result['gain']==1/64 and len(keys)==64
    summary=equal_fold_loss_summary([gain,gain])
    assert summary['mean_gain']==1/64 and summary['folds']==2
