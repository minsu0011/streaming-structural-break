"""Early integration never reads late labels or aliases normal eligibility."""
import src.next8
import numpy as np
import pytest
from tests.test_next8_movement_parents import feature_fixture
from tests.test_next8_joint_graph_models import selection_for
from src.next8.early_integration import early_prediction_rows,predict_early_head,normal_integration_relevant
from src.next8.joint_models import joint_design,prune_training_design
from src.next8.learning import fit_logistic
from src.next8.screening import sampled_rows


def test_early_prediction_cutoff_alias_and_no_target_access():
    cache,scope=feature_fixture();sel=selection_for(scope);tr=sampled_rows(cache,scope.training_ids,199)
    design=joint_design(cache,tr,sel,'M4');reduced,cols,_=prune_training_design(design)
    model=fit_logistic(reduced,cache.meta['target'][tr],scope)
    model.training.update({'early_training_cutoff':199,'original_input_names':list(design.names),'retained_input_indices':cols})
    cache.meta['time_online'][cache.meta['time_online']>=24]+=400
    class NoLabels:
        def __getitem__(self,key):raise AssertionError('Prediction used a label')
    cache.meta['target']=NoLabels();rows=early_prediction_rows(cache,scope)
    assert len(rows)>0 and (cache.meta['time_online'][rows]<=199).all()
    cross=np.full((len(rows),12),3,dtype=np.int8)
    p=predict_early_head(cache,scope,sel,model,'M4',rows,cross)
    assert len(p)==len(rows) and np.isfinite(p).all()
    late=np.flatnonzero(np.isin(cache.meta['dataset_id'],list(scope.validation_ids)) & (cache.meta['time_online']>199))[:1]
    with pytest.raises(ValueError,match='Late row'):predict_early_head(cache,scope,sel,model,'M4',late,cross[:1])


def test_normal_gate_controls_required_computation_only():
    receipts=[{'status':'COMPLETE','eligible':False} for _ in range(5)]
    assert normal_integration_relevant(receipts) is False
    receipts[3]['eligible']=True
    assert normal_integration_relevant(receipts) is True
    with pytest.raises(ValueError,match='All five'):normal_integration_relevant(receipts[:4])
