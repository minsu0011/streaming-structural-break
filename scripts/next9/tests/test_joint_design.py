import src.next9
import numpy as np
from src.next9.joint_design import assemble,alias_map
from src.next9.states import describe_history


class Bank:
    def __init__(self):
        self.values=np.sin(np.arange(50)*.2).astype(np.float32)
    def current(self,rows,columns):return self.values[rows,None]
    def history(self,rows,j):return np.stack([np.r_[np.zeros(16),self.values][r:r+16] for r in rows])
    def pre_state(self,rows,j):return describe_history(self.history(rows,j),.02,True)


def test_B_empty_preserves_eligible_F_and_history_controls():
    bank=Bank();rows=np.arange(20)
    a,an=assemble(bank,rows,[0],{}, {0:np.arange(10)}, {},'M1')
    b,bn=assemble(bank,rows,[0],{}, {0:np.arange(10)}, {},'M2')
    c,cn=assemble(bank,rows,[0],{}, {0:np.arange(10)}, {},'M3')
    assert a.shape[1]==5 and b.shape[1]>a.shape[1] and c.shape[1]>b.shape[1]
    aliases=alias_map([0],{}, {})
    assert aliases['M4']=='M3' and aliases['M5']=='M3'
    assert aliases['M2']=='M2'


def test_conditional_branch_abstains_outside_frozen_range():
    bank=Bank();rows=np.arange(20)
    specs={0:dict(pair=['S1_DIRECTION',2,0],low=10,high=11)}
    x,names=assemble(bank,rows,[0],specs,{0:np.arange(10)}, {},'M4')
    columns=[k for k,n in enumerate(names) if 'conditional' in n]
    np.testing.assert_array_equal(x[:,columns],0)
