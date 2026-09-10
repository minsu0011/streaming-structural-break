"""Refit a frozen DEV selection through the official training iterator contract."""
import hashlib,json
from pathlib import Path
import numpy as np
from src.features.config import CONFIG
from src.features.streaming import StreamingFeatureState,feature_names
from src.features.historical_calibration import CalibrationPolicy,HistoricalCalibratedState
from src.models.calibrated import CalibratedBundle
from src.models.learners import fit_candidate
from src.data.labels import labels_from_tau
from src.validation.research import contribution_weights
from src.utils.artifacts import write_json,sha256,utc_now

def sample_indices(length,policy):
    if policy in ('S0','S3','S3_sqrt'):return np.arange(length,dtype=np.int64)
    if policy=='S1':return np.unique(np.linspace(0,length-1,min(256,length),dtype=int)) if length else np.empty(0,dtype=np.int64)
    if policy=='S2':
        parts=[]
        for lo,hi in zip([0,10,25,50,100,200,400],[10,25,50,100,200,400,length]):
            hi=min(hi,length)
            if hi>lo:parts.append(np.unique(np.linspace(lo,hi-1,min(32,hi-lo),dtype=int)))
        return np.concatenate(parts) if parts else np.empty(0,dtype=np.int64)
    raise ValueError('Unfrozen training sampling policy')

def feature_state(historical,config,representation):
    from src.models.representations import make_state
    return make_state(historical,config,representation)

def train_frozen_candidate(datasets,model_directory_path,spec):
    roles={'FROZEN_PRIMARY':'PRIMARY','FROZEN_BACKUP':'BACKUP'}
    if spec.get('status') not in roles:raise RuntimeError('Training requires a frozen DEV-only candidate specification')
    if spec.get('use_fast_historical_init') and not spec.get('use_fused_inference'):raise ValueError('Fast initialization requires fused inference')
    candidate=spec['candidate'];allowed=set(map(int,spec['development_ids']));known=set(map(int,spec['known_training_ids']))
    if not allowed or not allowed<known:raise RuntimeError('Expected a nonempty DEV pool and a separate unopened seal')
    config=CONFIG.variant(**{k:tuple(v) if isinstance(v,list) else v for k,v in candidate.get('feature_changes',{}).items()})
    from src.models.representations import representation_feature_names
    representation=candidate.get('representation','base');policy=candidate['sampling'];names=representation_feature_names(representation,config)
    xs=[];ys=[];times=[];encountered=set();fitted_ids=[];skipped=0
    for sid,historical,online,tau in datasets:
        sid=int(sid)
        if sid in encountered or sid not in known:raise RuntimeError('Duplicate or unknown official training ID')
        encountered.add(sid)
        # Never inspect the sealed series' values, labels, lengths or arrays here.
        if sid not in allowed:skipped+=1;continue
        h=np.asarray(historical,dtype=np.float32);o=np.asarray(online,dtype=np.float32)
        state=feature_state(h,config,representation)
        selected=sample_indices(len(o),policy);keep=set(map(int,selected));matrix=np.empty((len(selected),len(names)),dtype=np.float32);position=0
        for t,point in enumerate(o):
            features=state.update_and_get(point)
            if t in keep:matrix[position]=features;position+=1
        target=labels_from_tau(len(o),tau)
        xs.append(matrix);ys.append(target[selected]);times.append(selected.astype(np.int32));fitted_ids.append(sid)
    if encountered!=known or set(fitted_ids)!=allowed:raise RuntimeError('Official training iterator does not match the frozen ID universe')
    features=np.concatenate(xs);target=np.concatenate(ys);time_online=np.concatenate(times)
    weights=contribution_weights(target,time_online,policy) if policy.startswith('S3') else None
    excluded=tuple(np.flatnonzero(np.ptp(features,axis=0)<=1e-12))
    if representation=='cross_order_ar4_ar8_equal':
        from src.models.cross_order import fit_cross_order,MODEL_NAME
        if candidate['model']!=MODEL_NAME or policy!='S3':raise ValueError('The fixed blend learner and weighting are immutable')
        model=fit_cross_order(features,target,families=candidate['families'],sample_weight=weights,params=candidate.get('params',{}),config=config)
    else:
        model=fit_candidate(candidate['model'],features,target,families=candidate['families'],sample_weight=weights,
            params=candidate.get('params',{}),config=config,excluded_columns=excluded)
    from src.models.representations import wrap_model,deployment_kind
    model=wrap_model(model,representation,config)
    directory=Path(model_directory_path);directory.mkdir(parents=True,exist_ok=True)
    temporary=directory/'model.pending.joblib';model.save(temporary);loaded=type(model).load(temporary)
    probe=features[:min(len(features),2048)]
    np.testing.assert_array_equal(model.predict(probe),loaded.predict(probe))
    temporary.replace(directory/'model.joblib')
    descriptor={'kind':deployment_kind(representation),'name':model.name,'feature_hash':model.feature_hash,
        'model_sha256':sha256(directory/'model.joblib'),'status':'FROZEN_DEV_'+roles[spec['status']]+'_REFIT','representation':representation,
        'candidate':candidate,'development_series_fitted':len(fitted_ids),'training_rows':len(target),'training_positive_rows':int(target.sum()),
        'seal_series_skipped_before_array_or_label_access':skipped,'seal_rows_fitted':0,'LOCAL_FINAL_SEAL_OPENED':False,
        'training_ids_sha256':hashlib.sha256(','.join(map(str,sorted(fitted_ids))).encode()).hexdigest(),
        'configuration_sha256':hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest(),'timestamp':utc_now(),
        'split_sha256':spec.get('split_sha256'),'training_source_sha256':sha256(Path(__file__)),
        'removed_training_constant_features':[names[j] for j in excluded],'use_fused_inference':bool(spec.get('use_fused_inference',False)),
        'use_fast_historical_init':bool(spec.get('use_fast_historical_init',False))}
    if descriptor['use_fused_inference']:
        from src.streaming.prepared import prepare_model,source_hashes
        descriptor.update(source_hashes(loaded,fast_initialization=descriptor['use_fast_historical_init']))
        prepared=prepare_model(loaded,fast_initialization=descriptor['use_fast_historical_init'])
        prepared.new_state(np.array([0.,1.],dtype=np.float32)).predict_one(0.)
    write_json(directory/'model.json',descriptor)
    return descriptor
