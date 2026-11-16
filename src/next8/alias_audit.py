"""Verify alias equivalence on complete heldout inputs before score release."""
import re
import numpy as np
import joblib
from src.next8.common import BASE,read_json,digest
from src.next8.cache import FeatureCache
from src.next8.features import DETAILED,UNKNOWN
from src.next8.data import rows_for
from src.next8.screening import procedure_scope
from src.next8.nested_auxiliary import early_scope,selected_cross_branch_needed
from src.next8.joint_models import joint_design,load_heldout_states


def assert_alias_coordinates(requested,donor,names):
    """Identical names at TRAIN do not alone prove UNKNOWN-gate equivalence."""
    a=[requested.names.index(n) for n in names];b=[donor.names.index(n) for n in names]
    np.testing.assert_array_equal(requested.ids,donor.ids)
    if not np.array_equal(requested.values[:,a],donor.values[:,b]):
        raise ValueError('Alias changes retained observable coordinates, including possible UNKNOWN gating')


def audit_scope_aliases(cache,directory):
    done=read_json(directory/'DONE.json');name=done['scope']['scope']
    match=re.fullmatch(r'outer_(\d)(?:_inner_(\d))?(_EARLY_T199)?',name)
    if match is None:raise ValueError('Unregistered model procedure scope')
    fold=int(match.group(1));inner=int(match.group(2)) if match.group(2) is not None else None;early=bool(match.group(3))
    scope=early_scope(procedure_scope(cache,fold,inner),early);assert scope.provenance()==done['scope']
    selection=read_json(BASE/'h_context_selection'/name/'SELECTION.json')
    graph=read_json(BASE/'parent_selection'/name/'GRAPH.json')
    assert selection['scope']==graph['scope']==scope.provenance()
    start=read_json(directory/'START.json')
    assert digest(BASE/'h_context_selection'/name/'SELECTION.json')==start['selection_sha256']
    assert digest(BASE/'parent_selection'/name/'GRAPH.json')==start['graph_sha256']
    for filename,sha in done['files_sha256'].items():assert digest(directory/filename)==sha,filename
    rows=rows_for(cache.meta,scope.validation_ids)
    if early:rows=rows[cache.meta['time_online'][rows]>=400]
    with np.load(directory/'PREDICTIONS.npz',allow_pickle=False) as a:
        np.testing.assert_array_equal(a['global_row'],rows)
        scope.admit_validation(np.unique(a['dataset_id']))
        for method,donor in done['model_aliases'].items():np.testing.assert_array_equal(a[method],a[donor])
    if selected_cross_branch_needed(selection,graph):
        cross,_=load_heldout_states(cache,scope,rows,BASE/'graph_states'/name)
    else:cross=np.full((len(rows),len(DETAILED)),UNKNOWN,dtype=np.int8)
    models={d:joblib.load(directory/(d+'.joblib')) for d in set(done['model_aliases'].values())}
    aliases={m:d for m,d in done['model_aliases'].items() if m!=d}
    for start in range(0,len(rows),50000):
        sl=slice(start,start+50000);designs={}
        for method,donor in aliases.items():
            for key in (method,donor):
                if key not in designs:designs[key]=joint_design(cache,rows[sl],selection,key,cross[sl])
            assert_alias_coordinates(designs[method],designs[donor],models[donor].names)
    return {'status':'PASS','scope':scope.provenance(),'heldout_rows':len(rows),'aliases_checked':aliases,
        'all_retained_coordinates_bitwise_equal_for_all_heldout_rows':True,
        'selection_stable_count':len(selection['stable_features']),'selection_variable_count':len(selection['variable_features']),
        'alias_prediction_arrays_bitwise_equal':True,'source_DONE_sha256':digest(directory/'DONE.json'),
        'labels_or_scores_used':False,'not_a_proof_for_all_unseen_possible_H_contexts':True,
        'alias_concern':'Same training names can hide different current-F UNKNOWN gating; complete actual heldout designs were checked'}


def audit_all_aliases():
    cache=FeatureCache();paths=sorted((BASE/'models').glob('*/DONE.json'))
    assert len(paths)>=25
    records=[audit_scope_aliases(cache,p.parent) for p in paths]
    return {'status':'PASS','procedures':len(records),'records':records,'source_file_changes':False,
        'future_reuse_requirement':'Re-run semantic alias gate for every new selected feature/state population; name equality alone is insufficient',
        'before_outer_score_release':True,'no_model_refit':True}
