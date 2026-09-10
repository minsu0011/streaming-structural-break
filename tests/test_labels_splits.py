import numpy as np
import pytest
from src.data.labels import labels_from_tau
from src.validation.splits import assignment,freeze_split,assert_development_ids
from src.validation.splits import ALGORITHM,SEED
from src.utils.artifacts import write_json

@pytest.mark.parametrize('tau,expected',[(None,[0,0,0]),(-1,[0,0,0]),(0,[1,1,1]),(1,[0,1,1]),(2,[0,0,1])])
def test_label_boundaries(tau,expected):
    np.testing.assert_array_equal(labels_from_tau(3,tau),expected)

@pytest.mark.parametrize('tau',[3,-2,1.5,np.nan,np.inf,True])
def test_invalid_tau(tau):
    with pytest.raises(ValueError):
        labels_from_tau(3,tau)

def test_frozen_split(tmp_path):
    frame=freeze_split(range(1000),tmp_path)
    assert freeze_split(reversed(range(1000)),tmp_path).equals(frame)
    with pytest.raises(RuntimeError):
        freeze_split(range(1001),tmp_path)
    assert set(frame.loc[frame.role=='DEVELOPMENT_POOL','fold']) == set(range(5))
    with pytest.raises(RuntimeError):
        assert_development_ids(frame.loc[frame.role=='LOCAL_FINAL_SEAL','dataset_id'])
    assert assignment(3)==assignment(np.int64(3))
    with pytest.raises(ValueError):
        freeze_split([3,3],tmp_path)

def test_pending_policy_allocation(tmp_path):
    write_json(tmp_path/'SPLIT_LOCK.json',{'status':'POLICY_FROZEN_IDS_PENDING','seed':SEED,'algorithm':ALGORITHM})
    frame=freeze_split(range(20),tmp_path)
    assert len(frame)==20
