"""Runtime case selection is fixed by original IDs and H lengths alone."""
import src.next8
from types import SimpleNamespace
import numpy as np
from src.next8.runtime_audit import fixed_cases
from tests.test_next8_streaming_runtime import actual_cache,fitted_fixture


def test_case_selection_does_not_use_online_length_or_labels():
    ids=np.array([9,2,3,8,1,6,5]);lengths=np.array([1000,1000,1600,1600,2100,4500,4500])
    data=SimpleNamespace(ids=ids,series={'historical_offsets':np.r_[0,np.cumsum(lengths)]})
    c=SimpleNamespace(data=data);scope=SimpleNamespace(validation_ids={1,2,3,5,6,8,9})
    result=fixed_cases(c,scope,2)
    assert [r['selected_ids'] for r in result]==[[2,9],[3,8],[1],[],[5,6]]
    assert sum(r['available'] for r in result)==7


def test_complete_execution_audit_with_unreadable_labels(actual_cache):
    from src.next8.runtime_audit import audit_case
    from src.next8.streaming_runtime import PreparedJoint
    from src.next8.joint_models import joint_design,model_design
    cache,h,online=actual_cache;heads={};rows=np.arange(len(online));saved={'global_row':rows}
    for method in ('M4','M5'):
        model,selection,graph,cross=fitted_fixture(cache,method)
        heads[method]=PreparedJoint(model,selection,method,{0:graph},resident=False)
        design=model_design(model,joint_design(cache,rows,selection,method,cross[rows]))
        saved[method]=model.predict_proba(design)[:,1].astype(np.float32)
    sid=int(cache.meta['dataset_id'][0])
    class Guard:
        def admit(self,ids,**kw):assert ids==[sid]
    class NoLabels:
        def __getitem__(self,key):raise AssertionError('Runtime audit read labels')
    old=cache.meta['target'];cache.meta['target']=NoLabels()
    cache.data=SimpleNamespace(guard=Guard(),id_to_index={sid:0},historical=h,online=online,
        series={'historical_offsets':np.array([0,len(h)]),'online_offsets':np.array([0,len(online)])})
    try:
        result,timing=audit_case(cache,sid,heads,None,None,saved,{'global_row':rows,'state':cross[rows]},False,timings=False)
        assert result['maximum_coordinate_error']==result['maximum_normalized_feature_error']==0
        assert result['exact_saved_prefix_chunk_future_suffix'] and not timing
    finally:cache.meta['target']=old
