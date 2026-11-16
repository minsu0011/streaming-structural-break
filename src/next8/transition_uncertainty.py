"""Conditional TRAIN table uncertainty, distinct from heldout forecast loss."""
import numpy as np
import joblib
from src.next8.common import BASE,read_json,digest
from src.next8.data import outer_scope
from src.next8.screening import sampled_rows
from src.next8.transitions import transition_inputs
from src.next8.movement import movement_target


def group_transition_counts(own,target,ids):
    own=np.asarray(own);target=np.asarray(target);ids=np.asarray(ids)
    known=(own<3)&(target<3);own=own[known];target=target[known];ids=ids[known]
    groups,code,count=np.unique(ids,return_inverse=True,return_counts=True)
    table=np.zeros((len(groups),9))
    np.add.at(table,(code,own*3+target),1./count[code])
    return groups,table


def table_uncertainty(group_counts,*,seed=8101604,draws=2000):
    group_counts=np.asarray(group_counts,dtype=np.float64);n=len(group_counts)
    if group_counts.shape!=(n,9) or n<64:raise ValueError('At least64 independent TRAIN groups required for table interval')
    rng=np.random.default_rng(seed);samples=[]
    point=group_counts.sum(axis=0).reshape(3,3)+.5;point/=point.sum(axis=1,keepdims=True)
    for start in range(0,draws,64):
        weight=rng.multinomial(n,np.full(n,1/n),size=min(64,draws-start))
        counts=(weight@group_counts).reshape(-1,3,3)+.5
        samples.append(counts/counts.sum(axis=2,keepdims=True))
    sampled=np.concatenate(samples)
    return {'point':point,'low':np.quantile(sampled,.025,axis=0),'high':np.quantile(sampled,.975,axis=0),
        'draws':draws,'groups':n,'seed':seed}


def actual_table_uncertainty(cache):
    scope=outer_scope(0,cache.data);j=0;h=1
    release=read_json(BASE/'PREDICTION_FREEZE_NEXT8.json');path=BASE/'transitions/outer_0/F000_h1/T3.joblib'
    assert digest(path)==release['files_sha256'][path.relative_to(BASE).as_posix()]
    model=joblib.load(path);rows=cache.forecast_rows(sampled_rows(cache,scope.training_ids),h)
    scope.admit_training(np.unique(cache.meta['dataset_id'][rows]))
    own,_=transition_inputs(cache,rows,j,[]);target=movement_target(cache,rows,j,h)['state']
    ids=cache.meta['dataset_id'][rows];groups,counts=group_transition_counts(own,target,ids)
    original=np.asarray([cache.data.guard.split.groups[int(s)] for s in groups])
    assert len(np.unique(original))==len(groups),'Per-series table weights and original dependency groups need explicit reconciliation'
    result=table_uncertainty(counts)
    error=float(np.max(np.abs(result['point']-model.own_table)));assert error<1e-12
    return {'status':'COMPLETE','scope':scope.provenance(),'feature':j,'horizon':h,'method':'T1 own table inside frozen T3',
        'point_probability':result['point'].tolist(),'ci_low':result['low'].tolist(),'ci_high':result['high'].tolist(),
        'groups':result['groups'],'draws':result['draws'],'seed':result['seed'],'model_point_parity_max_abs':error,
        'own_state_group_support':model.own_support.tolist(),'state_support_threshold':64,
        'source_model_sha256':digest(path),'source_group_count_array_not_exported_as_independent_rows':True,
        'interval_scope':'TRAIN whole-original-group bootstrap conditional on frozen feature/state definition; state/parent selection uncertainty not refitted',
        'not_heldout_accuracy_or_B_direction_uncertainty':True,'breakOccurred_labels_used':False,
        'unsupported_own_state_uses_prior_at_inference_even_if_raw_table_is_shown':True,
        'pointwise_nine_cell_intervals_not_joint_confidence_region':True}
