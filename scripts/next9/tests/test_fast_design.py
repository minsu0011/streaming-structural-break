import src.next9
from types import SimpleNamespace
import numpy as np
from src.next9.streaming import PrefixBank
from src.next9.joint_design import assemble
from src.next9.fast_design import frozen_design


def test_empty_rule_design_bitwise_random_and_unknown():
    rng=np.random.default_rng(9109)
    candidates=[72,81,89,59,85,77,64,51,1]
    maps={j:np.asarray([0,1,10,3,4,11,6,7,12,9,10,11,12]) for j in candidates}
    for available in [False,True]:
        seed=rng.uniform(-12,12,(16,153)).astype(np.float32)
        bank=PrefixBank(seed,np.full((3,153),.02),np.zeros(6),512 if available else 159)
        heads=SimpleNamespace(spec=dict(candidates=candidates,rule_specs={},run_maps={str(j):v.tolist() for j,v in maps.items()}),self_laws={})
        for t in range(257):
            bank.stage(rng.uniform(-12,12,153).astype(np.float32))
            expected,names=assemble(bank,[t],candidates,{},maps,{},'M5')
            heads.spec['input_names']=list(names)
            actual,anames=frozen_design(heads,bank)
            assert anames==names
            np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
            bank.commit()


def test_all_flat_and_runs_at_epsilon():
    for delta in [0.,float(np.float32(.02)),-float(np.float32(.02)),.1,-.1]:
        seed=np.tile((np.arange(16)*delta).astype(np.float32)[:,None],(1,153))
        bank=PrefixBank(seed,np.full((3,153),.02),np.zeros(6),512)
        bank.stage(np.full(153,.123,dtype=np.float32));maps={1:np.arange(13)}
        expected,names=assemble(bank,[0],[1],{},maps,{},'M5')
        heads=SimpleNamespace(spec=dict(candidates=[1],rule_specs={},run_maps={'1':list(range(13))},input_names=names),self_laws={})
        actual,_=frozen_design(heads,bank)
        np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))


def test_signed_zero_fixed_terms():
    bank=PrefixBank(np.zeros((16,153),np.float32),np.full((3,153),.02),np.zeros(6),512)
    bank.stage(np.full(153,-0.,dtype=np.float32))
    expected,names=assemble(bank,[0],[1],{},{1:np.arange(13)},{},'M5')
    heads=SimpleNamespace(spec=dict(candidates=[1],rule_specs={},run_maps={'1':list(range(13))},input_names=names),self_laws={})
    actual,_=frozen_design(heads,bank)
    np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
