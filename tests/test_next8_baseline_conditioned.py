import src.next8
import numpy as np
from tests.test_next8_movement_parents import feature_fixture
from tests.test_next8_joint_graph_models import selection_for
from src.next8.baseline_conditioned import fit_baseline_conditioned,baseline_designs
from src.next8.screening import sampled_rows
from src.next8.common import read_json


def test_conditional_BL_fit_only_reads_its_training_labels(tmp_path,monkeypatch):
    import src.next8.baseline_conditioned as module
    monkeypatch.setattr(module,'N_FEATURES',2)
    c,scope=feature_fixture();selection=selection_for(scope);original=c.meta['target'].copy()
    class TrainingOnly:
        def __getitem__(self,rows):
            assert np.all(c.meta['dataset_id'][rows]<128)
            return original[rows]
    c.meta['target']=TrainingOnly();tr=sampled_rows(c,scope.training_ids);va=sampled_rows(c,scope.validation_ids)
    rng=np.random.default_rng(391)
    fit_baseline_conditioned(c,scope,selection,tr,va,rng.uniform(.1,.9,len(tr)),rng.uniform(.1,.9,len(va)),
        np.full((len(tr),12),3,dtype=np.int8),np.full((len(va),12),3,dtype=np.int8),tmp_path)
    done=read_json(tmp_path/'F000/DONE.json')
    assert done['validation_labels_used'] is False
    assert done['model_aliases']['BL_CROSS_DIRECTION']=='BL_OWN_DIRECTION'
    assert read_json(tmp_path/'F001/DONE.json')['conditional_models_only_for_selected_variables'] is False


def test_BL_conditioned_tabular_has_exact_same_direction_inputs():
    c,scope=feature_fixture();rows=np.arange(70);sel=selection_for(scope)
    cross=np.tile(np.arange(12)%4,(len(rows),1)).astype(np.int8)
    x=baseline_designs(c,rows,0,np.full(len(rows),.5),sel,cross)
    assert x['BL_CROSS_DIRECTION'].names==x['BL_SAME_INPUT_TREE'].names
    np.testing.assert_array_equal(x['BL_CROSS_DIRECTION'].values,x['BL_SAME_INPUT_TREE'].values)
    assert not any(n.startswith('F_') or n.endswith('_times_F') for n in x['BL_STATE_ONLY'].names)
