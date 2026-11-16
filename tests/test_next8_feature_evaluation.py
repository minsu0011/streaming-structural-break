"""Real fit artifacts reach the atlas and conditional evaluator unchanged."""
import src.next8
from types import SimpleNamespace
import numpy as np
from tests.test_next8_movement_parents import feature_fixture
from tests.test_next8_joint_graph_models import selection_for
from src.next8.screening import sampled_rows
from src.next8.common import read_json,digest
from src.next8.direction_domains import training_cases
from src.next8.joint_models import orientation_states
from src.next8.h_context import HStateSpec,CONTEXT_MENU


def test_complete_feature_audit_uses_frozen_predictions(tmp_path,monkeypatch):
    import src.next8.orientation_refits as o
    import src.next8.baseline_conditioned as b
    import src.next8.feature_evaluation as e
    monkeypatch.setattr(o,'DETAILED',(0,));monkeypatch.setattr(b,'N_FEATURES',1);monkeypatch.setattr(e,'BASE',tmp_path)
    c,scope=feature_fixture();selection=selection_for(scope)
    selection['classifications']=[{'feature':j,'classification':'UNRESOLVED','signed_direction':[1,1,1],
        'repeated_conditional_rules':[]} for j in range(153)]
    c.series['dataset_id']=np.arange(256)
    c.data=SimpleNamespace(guard=SimpleNamespace(split=SimpleNamespace(groups={s:s for s in range(256)})))
    tr=sampled_rows(c,scope.training_ids);va=sampled_rows(c,scope.validation_ids)
    tc=np.full((len(tr),12),3,dtype=np.int8);vc=np.full((len(va),12),3,dtype=np.int8)
    op=tmp_path/'orientation_refits'/scope.name
    o.fit_orientation_refits(c,scope,selection,tc[:,:1],vc[:,:1],op)
    b.fit_baseline_conditioned(c,scope,selection,tr,va,np.full(len(tr),.5),np.full(len(va),.5),tc,vc,tmp_path/'baseline_conditioned'/scope.name)
    definitions=[]
    for method,cross in (('O2_OWN',False),('O3_GRAPH',True)):
        refit=read_json(op/'F000'/method/'DONE.json')
        cases=training_cases(c.current(tr,[0])[:,0],orientation_states(c,tr,0,selection),tc[:,0],c.meta['target'][tr],c.meta['dataset_id'][tr],refit,cross=cross)
        definitions.append({'feature':0,'orientation_model':method,'training_cases':cases})
    specs={kind:vars(HStateSpec.fit(c,scope,kind)) for kind in CONTEXT_MENU}
    release={'files_sha256':{p.relative_to(tmp_path).as_posix():digest(p) for p in tmp_path.rglob('PREDICTIONS.npz')}}
    result,arrays=e.evaluate_feature(c,scope,0,va,selection,specs,definitions,release)
    assert result['status']=='COMPLETE' and result['atlas_stride']==8 and result['selected_variable']
    assert len(result['BL80_conditional_information']['comparisons'])==5
    assert any(r['mode']=='O1_FIXED_NONLINEAR' for r in result['atlas'])
    assert result['conditional'][1]['status']=='INSUFFICIENT_TRAIN_SUPPORT_NO_REFIT'
    assert 'BL_C000_GAIN' in arrays and len(arrays['BL_C000_GAIN'])==128
