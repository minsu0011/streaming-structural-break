"""Immutable cohort receipts; interrupted incomplete attempts are kept intact."""
import hashlib
from pathlib import Path
import numpy as np
from src.next5.artifacts import ROOT,BASE,read_json,atomic_json,digest,utc_now,progress
from src.next5.synthetic import curves,cohort_key,reference_predictions,predict_model,combine_candidate


def update_input_hash(hasher,index,control,h,o,tau):
    hasher.update(np.asarray([index,int(control),tau,len(h),len(o)],dtype=np.int64).tobytes())
    hasher.update(np.asarray(h,dtype=np.float32).tobytes());hasher.update(np.asarray(o,dtype=np.float32).tobytes())


def cached(directory,identity):
    pointer=directory/'RESULTS.json'
    if not pointer.exists():return None
    record=read_json(pointer)
    if record['status']!='COMPLETE' or record['identity']!=identity:raise RuntimeError('Synthetic cohort identity changed')
    path=ROOT/record['prediction_file']
    if not path.resolve().is_relative_to(directory.resolve()) or digest(path)!=record['prediction_file_sha256']:
        raise RuntimeError('Synthetic cohort path/bytes changed')
    return record


def begin_attempt(directory,identity):
    directory.mkdir(parents=True,exist_ok=True);number=len(list(directory.glob('attempt_*')))+1
    out=directory/('attempt_'+str(number));out.mkdir(exist_ok=False)
    atomic_json(out/'START.json',{'created_utc':utc_now(),'identity':identity,'attempt':number,'prior_partial_attempts_preserved':number-1},immutable=True)
    return out


def finish(directory,out,identity,arrays,input_sha,extra):
    path=out/'PREDICTIONS.npz';np.savez(path,**arrays)
    record={'created_utc':utc_now(),'status':'COMPLETE','identity':identity,'prediction_file':path.relative_to(ROOT).as_posix(),
        'prediction_file_sha256':digest(path),'input_sha256':input_sha,'new_fits':0,'real_Shadow_rows':0,'LOCAL_SEAL_rows':0,**extra}
    atomic_json(out/'RESULTS.json',record,immutable=True);atomic_json(directory/'RESULTS.json',record,immutable=True)
    return record


def reference_cohort(models,proof,scope,fold,spec,menu,lock,execution_sha):
    key=cohort_key(spec);directory=BASE/'synthetic/references'/scope/('fold_'+str(fold))/key
    identity={'scope':scope,'fold':fold,'cohort':spec,'model_sources':proof,'menu_sha256':digest(BASE/'synthetic/MENU_LOCK.json'),'execution_sha256':execution_sha}
    prior=cached(directory,identity)
    if prior:return prior
    out=begin_attempt(directory,identity);count=spec['seeds']*(2 if spec['paired'] else 1);horizon=spec['horizon']
    roles=tuple(['C205','BL80']+(['B2'] if 'B2_augmented_J' in models else [])+(['D2'] if 'D2_expert' in models else []))
    arrays={role:np.empty((count,horizon),dtype=np.float64) for role in roles}
    arrays.update(synthetic_seed_index=np.empty(count,dtype=np.int32),control=np.empty(count,dtype=bool),tau=np.empty(count,dtype=np.int32))
    hasher=hashlib.sha256();last_index=-1;previous_h=previous_o=previous_p=None;reused=0
    for row,(index,control,h,o,tau) in enumerate(curves(spec,lock)):
        update_input_hash(hasher,index,control,h,o,tau)
        if index==last_index and np.array_equal(h,previous_h) and np.array_equal(o,previous_o):
            p=previous_p;reused+=1
        else:p=reference_predictions(models,h,o)
        for role in roles:arrays[role][row]=p[role]
        arrays['synthetic_seed_index'][row]=index;arrays['control'][row]=control;arrays['tau'][row]=tau
        last_index=index;previous_h=h;previous_o=o;previous_p=p
        if (row+1)%64==0:
            progress('Matched synthetic reference inference',scope+' fold '+str(fold)+' '+key+' '+str(row+1)+'/'+str(count))
            print('NEXT5 SYNTHETIC REFERENCE',scope,fold,key,row+1,count,flush=True)
    for model in models.values():model.validate()
    return finish(directory,out,identity,arrays,hasher.hexdigest(),{'curves':count,'horizon':horizon,'roles':roles,'identical_curve_predictions_reused':reused})


def candidate_cohort(model,registration,proof,scope,fold,spec,menu,lock,execution_sha,reference):
    cid=registration['experiment_id'];key=cohort_key(spec);directory=BASE/'synthetic/candidates'/cid/scope/('fold_'+str(fold))/key
    identity={'candidate':cid,'scope':scope,'fold':fold,'cohort':spec,'model_sources':proof,
        'menu_sha256':digest(BASE/'synthetic/MENU_LOCK.json'),'execution_sha256':execution_sha,'reference_prediction_sha256':reference['prediction_file_sha256']}
    prior=cached(directory,identity)
    if prior:return prior
    out=begin_attempt(directory,identity)
    with np.load(ROOT/reference['prediction_file'],allow_pickle=False) as source:
        ref={key:source[key] for key in ('BL80','C205','synthetic_seed_index','control','tau')}
    count=len(ref['tau']);horizon=spec['horizon'];arrays={name:np.empty((count,horizon),dtype=np.float64) for name in ('prediction','expert')}
    for name in ('synthetic_seed_index','control','tau'):arrays[name]=ref[name].copy()
    impulse=spec['kind']=='counterexample' and spec['scenario'] in ('O1','OB16')
    if impulse:arrays['features_at_impulse_lags']=np.empty((count,len(menu['impulse_lags']),len(model.names)),dtype=np.float32)
    hasher=hashlib.sha256();last_index=-1;previous_h=previous_o=previous_expert=previous_features=None;reused=0
    for row,(index,control,h,o,tau) in enumerate(curves(spec,lock)):
        assert index==ref['synthetic_seed_index'][row] and control==ref['control'][row] and tau==ref['tau'][row]
        update_input_hash(hasher,index,control,h,o,tau)
        if index==last_index and np.array_equal(h,previous_h) and np.array_equal(o,previous_o):
            expert=previous_expert;features=previous_features;reused+=1
        elif impulse:expert,features=predict_model(model,h,o,return_features=True)
        else:expert=predict_model(model,h,o);features=None
        arrays['expert'][row]=expert;arrays['prediction'][row]=combine_candidate(expert,ref['BL80'][row],ref['C205'][row],registration)
        if impulse:
            onset=lock['tau_cycle'][index%len(lock['tau_cycle'])];positions=onset+np.asarray(menu['impulse_lags']);assert positions.max()<len(o)
            arrays['features_at_impulse_lags'][row]=features[positions]
        last_index=index;previous_h=h;previous_o=o;previous_expert=expert;previous_features=features
        if (row+1)%64==0:
            progress('Fitted synthetic candidate inference',cid+' '+scope+' fold '+str(fold)+' '+key+' '+str(row+1)+'/'+str(count))
            print('NEXT5 SYNTHETIC CANDIDATE',cid,scope,fold,key,row+1,count,flush=True)
    assert hasher.hexdigest()==reference['input_sha256'];model.validate()
    return finish(directory,out,identity,arrays,hasher.hexdigest(),{'curves':count,'horizon':horizon,
        'exact_generator_and_reference_alignment':True,'identical_curve_predictions_reused':reused,'impulse_feature_lags_saved':impulse})
