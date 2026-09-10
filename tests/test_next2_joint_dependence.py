import numpy as np
import pytest
from src.next2.joint_dependence import JointState
from src.next2.complement_bank import ComplementState
from src.next.serial_variance_evidence import SerialVarianceState,SerialVarianceConfig
from src.next.channel_evidence import CHANNEL_STATS


@pytest.mark.parametrize('summary',['ewma32','bayes025'])
def test_joint_features_equal_separately_audited_components(summary):
    rng=np.random.default_rng(482)
    h=rng.standard_t(5,2048)
    o=rng.normal(size=1100).astype(np.float32)
    o[700:]*=.6
    parent=SerialVarianceState(h,SerialVarianceConfig(max_age=512)).replay(o)[:,:26]
    extra=ComplementState(h,{'family':'dependence'}).replay(o)
    columns=[13*j+CHANNEL_STATS.index(summary) for j in range(4)]
    result=JointState(h,{'summary':summary}).replay(o)
    np.testing.assert_array_equal(result,np.c_[parent,extra[:,columns]])
    np.testing.assert_array_equal(result[:500],JointState(h,{'summary':summary}).replay(o[:500]))
