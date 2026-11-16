"""Observable pre-current state issued by the actual sparse transition model."""
from pathlib import Path
import numpy as np
import joblib
from src.next8.common import BASE,atomic,read_json,digest,utc_now,activity
from src.next8.features import DETAILED,UNKNOWN
from src.next8.data import rows_for
from src.next8.cache import immutable_npz
from src.next8.screening import sampled_rows
from src.next8.movement import movement_target
from src.next8.forecasting import parents_at,validate_graph_scope
from src.next8.transitions import FactorizedTransition,transition_inputs

MIN_PROBABILITY=.60
MIN_PROBABILITY_MARGIN=.15


def predict_graph_state(cache,rows,j,model):
    rows=np.asarray(rows,dtype=np.int64)
    state=np.full(len(rows),UNKNOWN,dtype=np.int8)
    probability=np.tile([0.,0.,0.,1.],(len(rows),1)).astype(np.float32)
    active=np.zeros(len(rows),dtype=np.int8)
    valid=np.flatnonzero(cache.meta['time_online'][rows]>=1)
    for start in range(0,len(valid),50000):
        at=valid[start:start+50000];origin=rows[at]-1
        assert np.array_equal(cache.meta['dataset_id'][origin],cache.meta['dataset_id'][rows[at]])
        own,other=transition_inputs(cache,origin,j,list(model.parents))
        p,evidence=model.predict(own,other)
        probability[at]=p;active[at]=evidence['active_parent_factors']
        order=np.sort(p[:,:3],axis=1)
        sure=(p[:,3]==0)&(order[:,-1]>=MIN_PROBABILITY)&((order[:,-1]-order[:,-2])>=MIN_PROBABILITY_MARGIN)
        sure &= evidence['active_parent_factors']>=1
        state[at[sure]]=np.argmax(p[sure,:3],axis=1)
    return state,probability,active


def fit_graph_states(cache,scope,graph,directory,*,early=False,reuse_directory=None):
    validate_graph_scope(graph,scope)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if (directory/'DONE.json').exists():
        done=read_json(directory/'DONE.json');assert done['scope']==scope.provenance()
        for name,sha in done['files_sha256'].items():assert digest(directory/name)==sha
        return done
    if any(p.name!='START.json' for p in directory.iterdir()):raise RuntimeError('Partial graph-state output preserved')
    cutoff=199 if early else None
    tr=cache.forecast_rows(sampled_rows(cache,scope.training_ids,cutoff),1,cutoff)
    va=rows_for(cache.meta,scope.validation_ids)
    scope.admit_training(np.unique(cache.meta['dataset_id'][tr]));scope.admit_validation(np.unique(cache.meta['dataset_id'][va]))
    state=np.full((len(va),len(DETAILED)),UNKNOWN,dtype=np.int8)
    probability=np.tile([0.,0.,0.,1.],(len(va),len(DETAILED),1)).astype(np.float32)
    active=np.zeros((len(va),len(DETAILED)),dtype=np.int8)
    records=[];files={}
    for k,j in enumerate(DETAILED):
        parents=parents_at(graph,j,1)
        if not parents:
            records.append({'feature':j,'status':'NO_H1_PARENT_UNKNOWN_FALLBACK'});continue
        with activity(f'Cross-fitted factorized state {scope.name} F{j:03d}','GRAPH_STATE','B2_STATE_OBSERVABILITY'):
            donor=Path(reuse_directory)/f'F{j:03d}_h1' if reuse_directory is not None else None
            reused=None
            if donor is not None and (donor/'DONE.json').exists():
                receipt=read_json(donor/'DONE.json')
                assert receipt['scope']==scope.provenance() and receipt['parents']==parents
                assert digest(donor/'T3.joblib')==receipt['files_sha256']['T3.joblib']
                model=joblib.load(donor/'T3.joblib')
                assert model.training['training_rows']==int(np.sum(movement_target(cache,tr,j,1)['state']<UNKNOWN))
                reused={'source':(donor/'T3.joblib').relative_to(BASE).as_posix(),'sha256':digest(donor/'T3.joblib')}
            else:
                own,other=transition_inputs(cache,tr,j,parents)
                target=movement_target(cache,tr,j,1)['state']
                model=FactorizedTransition.fit(own,other,target,cache.meta['dataset_id'][tr],parents,scope)
            assert tuple(parents)==model.parents
            state[:,k],probability[:,k],active[:,k]=predict_graph_state(cache,va,j,model)
            path=directory/f'F{j:03d}_T3.joblib';joblib.dump(model,path);files[path.name]=digest(path)
            records.append({'feature':j,'status':'COMPLETE','parents':parents,'training':model.training,
                'parent_selection_scope':scope.provenance(),'model_sha256':digest(path),
                'reused_transition_model':reused,'new_fit':reused is None})
    immutable_npz(directory/'STATE_PREDICTIONS.npz',global_row=va,dataset_id=cache.meta['dataset_id'][va],
        feature=np.asarray(DETAILED,dtype=np.int32),state=state,probability=probability,active_factors=active)
    files['STATE_PREDICTIONS.npz']=digest(directory/'STATE_PREDICTIONS.npz')
    done={'status':'COMPLETE','timestamp':utc_now(),'scope':scope.provenance(),'models':records,
        'files_sha256':files,'state_estimator':'FACTORISED_T3_PRE_CURRENT_V1',
        'prediction_origin':'At t-1, own physical state at t-1 and parent states at t-2; no F[t]',
        'minimum_probability':MIN_PROBABILITY,'minimum_probability_margin':MIN_PROBABILITY_MARGIN,
        'minimum_supported_parent_factors':1,'unknown_fallback':'No cross contribution; preserve self/H and stable model branches',
        'probabilities_not_calibrated_coverage_claim':True,'forecast_not_observed_event':True,
        'complete_fit_and_selection_exclude_prediction_groups':True,'outer_score_released':False}
    atomic(directory/'DONE.json',done)
    return done
