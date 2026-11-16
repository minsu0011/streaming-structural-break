import src.next9
import numpy as np
from src.next9.cross import proposals,qualified_horizon,design
from src.next8.features import mechanical_controls
from src.next9.states import describe_history


def test_cross_budget_and_zero_branch():
    candidates=[72,81,89,59,85,77,64,51,1]
    assert proposals(candidates,{},[])==[]
    edges=proposals(candidates,{j:{} for j in candidates[:6]},candidates[:6])
    assert len(edges)<=8
    for j in candidates:
        assert sum(d==j for d,p in edges)<=2
    assert all(p not in mechanical_controls(d) for d,p in edges)


def test_cross_requires_same_horizon_and_exact_mechanical_control():
    def contrast(gain,lo):return dict(groups=100,mean_gain=gain,ci_low=lo)
    rows=[]
    for split in range(3):
        rows.append(dict(next_state_contrast=contrast(.1,.05),forecast_contrasts=[dict(horizon=h,D3=contrast(.1,.05),D5=contrast(.1,.05 if k==split else -.1)) for k,h in enumerate([1,4,16])]))
    assert qualified_horizon(rows) is None
    for r in rows:r['forecast_contrasts'][1]['D5']['ci_low']=.01
    assert qualified_horizon(rows)==4


def test_cross_parent_current_and_suffix_never_enter_design():
    class Bank:
        def __init__(self,v):
            self.v=v;self.series=dict(context=np.zeros((1,6)));self.series_index=np.zeros(len(v),dtype=int)
        def history(self,rows,j):
            return np.stack([np.r_[np.zeros(16),self.v[:,j]][r:r+16] for r in rows])
        def pre_state(self,rows,j):return describe_history(self.history(rows,j),.02,True)
    v=np.random.default_rng(9101604).normal(size=(40,153)).astype(np.float32)
    d,p=proposals([72,81,89,59,85,77,64,51,1],{72:{}},[72])[0]
    changed=v.copy();changed[20:]=100
    np.testing.assert_array_equal(design(Bank(v),np.arange(21),d,[p]),design(Bank(changed),np.arange(21),d,[p]))
