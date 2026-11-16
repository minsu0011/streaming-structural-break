import src.next8
import numpy as np
from tests.test_next8_movement_parents import feature_fixture
from tests.test_next8_joint_graph_models import selection_for
from src.next8.orientation_refits import fit_orientation_refits,fit_conditional_refit,designs
from src.next8.screening import sampled_rows


def test_direction_refit_does_not_read_validation_labels(tmp_path,monkeypatch):
    import src.next8.orientation_refits as module
    monkeypatch.setattr(module,'DETAILED',(0,))
    c,scope=feature_fixture();selection=selection_for(scope);original=c.meta['target'].copy()
    class TrainingOnly:
        def __getitem__(self,rows):
            if np.any(c.meta['dataset_id'][rows]>=128):raise AssertionError('Validation labels entered direction fitting')
            return original[rows]
    c.meta['target']=TrainingOnly()
    tr=sampled_rows(c,scope.training_ids);va=sampled_rows(c,scope.validation_ids)
    fit_orientation_refits(c,scope,selection,np.full((len(tr),1),3,dtype=np.int8),np.full((len(va),1),3,dtype=np.int8),tmp_path)
    from src.next8.common import read_json
    done=read_json(tmp_path/'F000/DONE.json')
    assert done['conditional']['O3_GRAPH']['status']=='INSUFFICIENT_SUPPORT'
    assert done['validation_labels_used'] is False
    assert done['conditional']['O2_OWN']['status']=='COMPLETE'


def test_cross_orientation_control_removes_only_cross_F_interactions():
    c,_=feature_fixture();rows=np.arange(30);own=np.arange(30)%3;cross=(np.arange(30)+1)%3
    x=designs(c,rows,0,own,cross)
    removed=set(x['DIRECTION'].names)-set(x['FIXED_CONTROL'].names)
    assert removed=={'cross_state_0_times_F','cross_state_2_times_F'}
    assert 'own_state_0_times_F' in x['FIXED_CONTROL'].names
    assert 'cross_state_0' in x['FIXED_CONTROL'].names
    assert not any(n.startswith('F_') or n.endswith('_times_F') for n in x['STATE_ONLY'].names)
    assert x['DIRECTION'].names==x['TABULAR'].names
    np.testing.assert_array_equal(x['DIRECTION'].values,x['TABULAR'].values)
