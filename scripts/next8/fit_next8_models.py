"""Fit the frozen M menu using fully excluded-group graph-state inputs."""
from pathlib import Path
import sys
import argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next8
import numpy as np
from src.next8.common import BASE,read_json,atomic,digest,utc_now,note
from src.next8.cache import FeatureCache
from src.next8.features import DETAILED,UNKNOWN
from src.next8.data import rows_for
from src.next8.screening import procedure_scope,selection_splits,sampled_rows
from src.next8.nested_auxiliary import early_scope,selected_cross_branch_needed
from src.next8.joint_models import fit_joint_models,load_state_partition,load_heldout_states


def prepare_inputs(cache,fold,inner,early,scope,selection,graph):
    assert selection['scope']==graph['scope']==scope.provenance()
    tr=sampled_rows(cache,scope.training_ids,199 if early else None)
    va=rows_for(cache.meta,scope.validation_ids)
    if early:va=va[cache.meta['time_online'][va]>=400]
    needed=selected_cross_branch_needed(selection,graph)
    if not needed:
        return np.full((len(tr),len(DETAILED)),UNKNOWN,dtype=np.int8),np.full((len(va),len(DETAILED)),UNKNOWN,dtype=np.int8),{
            'scope':scope.provenance(),'cross_branch_needed':False,'reason':'No selected feature has a supported h1 cross parent in this training scope',
            'no_in_sample_state_inputs':True,'no_unused_leaf_fits_required':True}
    splits=[early_scope(s,early) for s in selection_splits(cache,fold,inner)]
    if inner is not None:
        parent=BASE/'meta_state_inputs'/scope.name;done=read_json(parent/'DONE.json');assert done['required']
        parts=[(splits[r['index']],parent/r['state_directory']) for r in done['parts']]
    elif early:
        parent=BASE/'early_state_inputs'/scope.name;done=read_json(parent/'DONE.json');assert done['required']
        parts=[(splits[r['index']],parent/r['state_directory']) for r in done['parts']]
    else:parts=[(s,BASE/'graph_states'/s.name) for s in splits]
    train,receipts=load_state_partition(cache,scope,tr,parts)
    valid,validation=load_heldout_states(cache,scope,va,BASE/'graph_states'/scope.name)
    return train,valid,{'scope':scope.provenance(),'cross_branch_needed':True,'training_state_parts':receipts,
        'validation_state_model':validation,'all_state_fit_and_graph_selection_exclude_prediction_group':True,
        'all12_forecast_channels_generated_before_final_feature_subset_used':True}


def main(fold,inner,early):
    assert read_json(BASE/'FINAL_MODEL_TEST_GATE_NEXT8.json')['status']=='PASS'
    lock_path=BASE/'FINAL_MODEL_IMPLEMENTATION_LOCK_NEXT8.json';lock=read_json(lock_path)
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    cache=FeatureCache();scope=early_scope(procedure_scope(cache,fold,inner),early)
    sp=BASE/'h_context_selection'/scope.name;sd=read_json(sp/'DONE.json')
    assert digest(sp/'SELECTION.json')==sd['selection_sha256'];selection=read_json(sp/'SELECTION.json')
    gp=BASE/'parent_selection'/scope.name;gd=read_json(gp/'DONE.json')
    assert digest(gp/'GRAPH.json')==gd['graph_sha256'];graph=read_json(gp/'GRAPH.json')
    directory=BASE/'models'/scope.name;directory.mkdir(parents=True,exist_ok=True)
    contract={'scope':scope.provenance(),'selection_sha256':digest(sp/'SELECTION.json'),
        'graph_sha256':digest(gp/'GRAPH.json'),'implementation_lock_sha256':digest(lock_path),
        'early_training_cutoff':199 if early else None}
    if (directory/'START.json').exists():
        old=read_json(directory/'START.json')
        for k,v in contract.items():assert old[k]==v,k
    else:atomic(directory/'START.json',{'status':'FROZEN','timestamp':utc_now(),**contract})
    if (directory/'DONE.json').exists():
        done=read_json(directory/'DONE.json')
        for name,sha in done['files_sha256'].items():assert digest(directory/name)==sha
        raise RuntimeError('Completed final models cannot be silently rerun')
    train,valid,receipt=prepare_inputs(cache,fold,inner,early,scope,selection,graph)
    if (directory/'STATE_INPUT_PROVENANCE.json').exists():assert read_json(directory/'STATE_INPUT_PROVENANCE.json')==receipt
    else:atomic(directory/'STATE_INPUT_PROVENANCE.json',receipt)
    methods=None if inner is None else ['M4','M5']
    fit_joint_models(cache,scope,selection,directory,train,valid,early=early,methods=methods)
    note('Frozen final model predictions complete',scope.name+'; no heldout y read and no scores released','B4_MOVEMENT_VS_DISCRIMINATION')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=range(5),required=True)
    p.add_argument('--meta-inner',type=int,choices=range(3));p.add_argument('--early',action='store_true')
    a=p.parse_args();main(a.fold,a.meta_inner,a.early)
