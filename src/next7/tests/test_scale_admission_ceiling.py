import numpy as np
import pytest
from src.next7.event import event_step,new_event_args
from src.next7.null_growth import admission_step,new_admission_args


def raw_at_scale_ceiling(scale_cut,core_ratio=0.):
    core=np.zeros(20);core[1]=core_ratio*2.;core[11]=4 if core_ratio>1 else 0;core[12]=4.
    scale=np.zeros(22);scale[0]=1;scale[1]=4;scale[2]=-4.;scale[7:10]=[4.,-1.,4.]
    return event_step(core,scale,new_event_args(2.,scale_cut,0,1,True,True,True)).copy()


@pytest.mark.parametrize('scale_cut',[2.2079362843103825,2.0409484800482542,1.9041449324453819])
def test_scale_ceiling_equal_cut_rejected_but_one_float_below_admits(scale_cut):
    cap=4/scale_cut
    raw=raw_at_scale_ceiling(scale_cut)
    assert raw[2]==cap and raw[22]==8 and raw[18]==1
    for cut,expected in ((cap,False),(np.nextafter(cap,0),True),(cap+.1,False)):
        args=new_admission_args({'admission_cuts':[cut]*9,'reference_valid':True,'unsupported_cut':1e12})
        actual=admission_step(raw,args)
        assert bool(actual[40])==expected


def test_core_can_admit_when_scale_cannot_and_winner_is_core():
    scale_cut=2.0409484800482542;cut=4/scale_cut+.1
    raw=raw_at_scale_ceiling(scale_cut,core_ratio=cut+.1)
    args=new_admission_args({'admission_cuts':[cut]*9,'reference_valid':True,'unsupported_cut':1e12})
    actual=admission_step(raw,args)
    assert actual[40]==1 and actual[37]==0 and raw[22]==0
