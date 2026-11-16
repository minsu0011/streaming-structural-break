"""Necessary auxiliary-only nesting for state inputs to complete meta models."""
import hashlib
from pathlib import Path
from src.next8.common import atomic,read_json,digest,utc_now
from src.next8.data import FitScope
from src.next8.features import DETAILED
from src.next8.screening import selection_splits,procedure_scope
from src.next8.movement import parent_selection_split,select_parent_graph
from src.next8.forecasting import parents_at
from src.next8.graph_states import fit_graph_states


def subdivide_scope(parent,group_mapping):
    groups={}
    for sid in sorted(parent.training_ids):groups.setdefault(group_mapping[sid],[]).append(sid)
    if len(groups)<3:raise ValueError('At least three independent training groups')
    def key(group):return hashlib.sha256(f'NEXT8_AUXILIARY_LEAF_V1|{parent.name}|{group}'.encode()).hexdigest()
    assignment={g:k%3 for k,g in enumerate(sorted(groups,key=key))}
    result=[]
    for k in range(3):
        valid=frozenset(s for g,members in groups.items() if assignment[g]==k for s in members)
        result.append(FitScope(parent.training_ids-valid,valid,parent.validation_ids|parent.outer_excluded_ids,
                               parent.name+f'_auxiliary_inner_{k}'))
    return result


def early_scope(scope,early):
    if not early:return scope
    return FitScope(scope.training_ids,scope.validation_ids,scope.outer_excluded_ids,scope.name+'_EARLY_T199')


def selected_cross_branch_needed(selection,graph):
    # This only skips unused calculations. If needed, every one of the fixed12
    # channels and the complete fixed12 graph selection is generated below.
    return any(parents_at(graph,int(j),1) for j in selection['variable_features'])


def select_graph_nested(cache,parent,directory,*,early=False):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if (directory/'DONE.json').exists():
        done=read_json(directory/'DONE.json');assert done['scope']==parent.provenance()
        assert digest(directory/'GRAPH.json')==done['graph_sha256']
        return read_json(directory/'GRAPH.json')
    if not (directory/'START.json').exists():atomic(directory/'START.json',{
        'status':'FROZEN','timestamp':utc_now(),'scope':parent.provenance(),'training_cutoff':199 if early else None,
        'subdivision':'NEXT8_AUXILIARY_LEAF_V1 deterministic original dependency groups',
        'parent_pool':list(DETAILED),'breakOccurred_used':False})
    panels=[];corrections=[]
    for k,scope in enumerate(subdivide_scope(parent,cache.data.guard.split.groups)):
        cache.data.guard.partition(scope.training_ids,scope.validation_ids|scope.outer_excluded_ids)
        p=directory/f'split_{k}'
        if (p/'DONE.json').exists():
            done=read_json(p/'DONE.json');assert done['scope']==scope.provenance()
            for name,sha in done['records_sha256'].items():assert digest(p/name)==sha
            records=[read_json(p/name) for name in done['records_sha256']]
            adjustment=read_json(p/'MULTIPLICITY.json')['tests']
        else:records,adjustment=parent_selection_split(cache,scope,p,cutoff=199 if early else None)
        panels.append(records);corrections.append(adjustment)
    graph=select_parent_graph(panels,corrections,parent)
    atomic(directory/'GRAPH.json',graph)
    atomic(directory/'DONE.json',{'status':'COMPLETE','timestamp':utc_now(),'scope':parent.provenance(),
        'graph_sha256':digest(directory/'GRAPH.json'),'all_predicted_groups_excluded_from_parent_selection':True,
        'all12_fixed_channels_considered_even_if_final_model_uses_subset':True,'breakOccurred_used':False})
    return graph


def fit_meta_leaf_states(cache,fold,inner,selection,graph,directory,*,early=False):
    parent=early_scope(procedure_scope(cache,fold,inner),early)
    assert selection['scope']==graph['scope']==parent.provenance()
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if (directory/'DONE.json').exists():return read_json(directory/'DONE.json')
    required=selected_cross_branch_needed(selection,graph)
    receipt={'scope':parent.provenance(),'required':required,'early_training_cutoff':199 if early else None,
        'reason':'Selected M5 cross-state branch requires full auxiliary-only nesting' if required else 'No selected feature has a supported h1 parent; cross-state inputs unused and UNKNOWN',
        'state_channel_count':len(DETAILED),'parts':[]}
    if required:
        for k,scope in enumerate(selection_splits(cache,fold,inner)):
            scope=early_scope(scope,early)
            p=directory/f'leaf_{k}'
            selected=select_graph_nested(cache,scope,p/'parent_selection',early=early)
            fit_graph_states(cache,scope,selected,p/'graph_states',early=early)
            receipt['parts'].append({'index':k,'scope':scope.provenance(),
                'graph_directory':f'leaf_{k}/parent_selection','state_directory':f'leaf_{k}/graph_states',
                'state_done_sha256':digest(p/'graph_states/DONE.json')})
    atomic(directory/'DONE.json',{'status':'COMPLETE','timestamp':utc_now(),**receipt})
    return read_json(directory/'DONE.json')


def fit_early_outer_state_parts(cache,fold,selection,graph,directory):
    """Early outer M5 needs train<=199 state OOF from original inner groups."""
    parent=early_scope(procedure_scope(cache,fold),True)
    assert selection['scope']==graph['scope']==parent.provenance()
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if (directory/'DONE.json').exists():return read_json(directory/'DONE.json')
    required=selected_cross_branch_needed(selection,graph);parts=[]
    if required:
        for k,scope in enumerate(selection_splits(cache,fold)):
            scope=early_scope(scope,True);p=directory/f'inner_{k}'
            selected=select_graph_nested(cache,scope,p/'parent_selection',early=True)
            fit_graph_states(cache,scope,selected,p/'graph_states',early=True)
            parts.append({'index':k,'scope':scope.provenance(),'state_directory':f'inner_{k}/graph_states',
                          'state_done_sha256':digest(p/'graph_states/DONE.json')})
    atomic(directory/'DONE.json',{'status':'COMPLETE','timestamp':utc_now(),'scope':parent.provenance(),
        'required':required,'parts':parts,'early_training_cutoff':199,
        'no_late_feature_or_future_target_used':True,'auxiliary_only_no_break_labels':True})
    return read_json(directory/'DONE.json')
