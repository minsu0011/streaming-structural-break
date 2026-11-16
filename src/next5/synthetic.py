"""Matched ResearchPool fitted artifacts and unchanged synthetic cohort paths."""
import math
import numpy as np
from src.next5.artifacts import ROOT,BASE,read_json,digest
from src.next5.model import FeatureModel,feature_module
from src.next2.counterexamples import generate
from src.next2.cross_background_scale import series as crossed_series


def cohort_series(kind,scenario,index,horizon,control,lock):
    if kind not in ('calibration','counterexample','long_null'):raise ValueError('Fixed synthetic partition required')
    partition={'calibration':0,'counterexample':1,'long_null':2}[kind]
    if scenario.startswith('CB_N3_'):
        if kind!='counterexample':raise ValueError('Fixed crossed backgrounds are counterexamples only')
        factor={'CB_N3_070':.7,'CB_N3_050':.5}[scenario]
        return crossed_series(lock,'N3',factor,index,control=control,horizon=horizon)
    if scenario=='OB16':
        if kind!='counterexample':raise ValueError('Fixed burst is a paired counterexample only')
        h,o,tau=generate('O1',index,partition=1,horizon=horizon,control=True,lock=lock)
        if not control:
            start=int(lock['tau_cycle'][index%len(lock['tau_cycle'])]);stop=min(start+16,len(o))
            o[start:stop]+=np.asarray([8.*(1 if (index+j)%2==0 else -1) for j in range(stop-start)],dtype=np.float32)
        return h,o,-1
    return generate(scenario,index,partition=partition,horizon=horizon,control=control,lock=lock)


def load_reference(scope,fold):
    models={};receipts={}
    kinds=('J102','C205','B2_augmented_J','D2_expert') if scope=='full' else ('J102','C205')
    for kind in kinds:
        out=BASE/'references/fits/PRIMARY'/scope/kind/('fold_'+str(fold));receipt=read_json(out/'RESULTS.json')
        path=out/'model.joblib'
        if digest(path)!=receipt['model_sha256']:raise RuntimeError('Matched reference bytes changed')
        models[kind]=FeatureModel.load(path);receipts[kind]={'model_sha256':digest(path),'result_sha256':digest(out/'RESULTS.json')}
    ids={m.training['training_ids_sha256'] for m in models.values()}
    if len(ids)!=1:raise RuntimeError('Synthetic reference training groups differ')
    return models,receipts


def load_candidate(cid,scope,fold):
    out=BASE/'experiments'/cid/scope/('fold_'+str(fold));receipt=read_json(out/'RESULTS.json');path=out/'model.joblib'
    if digest(path)!=receipt['model_sha256']:raise RuntimeError('Fitted candidate bytes changed')
    registration=read_json(BASE/'registrations'/(cid+'.json'))
    if digest(BASE/'registrations'/(cid+'.json'))!=receipt['registration_sha256']:raise RuntimeError('Candidate recipe changed')
    model=FeatureModel.load(path)
    return model,registration,{'model_sha256':digest(path),'result_sha256':digest(out/'RESULTS.json'),'registration_sha256':receipt['registration_sha256']}


def predict_model(model,historical,online,*,return_features=False):
    # Caller validates immutable model and feature source before and after each
    # cohort; avoid repeating disk hashes at every synthetic H initialization.
    state=feature_module(model.module_name).make_state(historical,model.settings)
    features=state.replay(online);prediction=model.predict(features)
    if prediction.shape!=(len(online),) or not np.isfinite(prediction).all():raise RuntimeError('Invalid synthetic inference output')
    return (prediction,features) if return_features else prediction


def reference_predictions(models,h,o):
    p={name:predict_model(model,h,o) for name,model in models.items()}
    result={'C205':p['C205'],'BL80':.8*p['J102']+.2*p['C205']}
    if 'B2_augmented_J' in p:result['B2']=.8*p['B2_augmented_J']+.2*p['C205']
    if 'D2_expert' in p:result['D2']=.9*result['BL80']+.1*p['D2_expert']
    return result


def combine_candidate(expert,baseline,dependence,registration):
    weights=registration['combination']
    return weights.get('ResearchPool_BL80',0.)*baseline+weights.get('C205',0.)*dependence+weights['new_model']*expert


def background_mapping(lock):
    result={}
    for spec in lock['scenarios']:result.setdefault((spec['phi_before'],spec['arch_before'],spec['t_df_before']),spec['id'])
    return result


def scenario_background(scenario,lock):
    if scenario.startswith('CB_N3_'):scenario='N3'
    if scenario=='OB16':scenario='O1'
    spec=next(s for s in lock['scenarios'] if s['id']==scenario)
    return (spec['phi_before'],spec['arch_before'],spec['t_df_before'])


def cohort_menu(menu,lock,*,whole_research=False):
    sizes=menu['frozen_whole_ResearchPool_model_sizes'] if whole_research else menu['fold_model_sizes'];result=[]
    for scenario in background_mapping(lock).values():
        result.append({'kind':'calibration','scenario':scenario,'seeds':sizes['calibration'],'horizon':1024,'paired':False,'control':True})
    for scenario in menu['counterexamples']+menu['additional_counterexamples']:
        result.append({'kind':'counterexample','scenario':scenario,'seeds':sizes['evaluation'],
            'horizon':menu['impulse_horizon'] if scenario in ('O1','OB16') else menu['counterexample_horizon'],'paired':True,'control':False})
    for scenario in menu['long_null_regimes']:
        result.append({'kind':'long_null','scenario':scenario,'seeds':sizes['long_null'],'horizon':max(menu['long_null_horizons']),'paired':False,'control':False})
    return result


def cohort_key(spec):return spec['kind']+'__'+spec['scenario']


def curves(spec,lock):
    for index in range(spec['seeds']):
        for control in ((False,True) if spec['paired'] else (spec['control'],)):
            h,o,tau=cohort_series(spec['kind'],spec['scenario'],index,spec['horizon'],control,lock)
            yield index,control,h,o,tau


def threshold(maxima,alpha):
    values=np.asarray(maxima,dtype=float)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all() or not 0<alpha<1:raise ValueError('Valid independent null maxima required')
    rank=math.ceil((len(values)+1)*(1-alpha))
    return float(np.sort(values)[rank-1]) if rank<=len(values) else 1.


def alarm_summary(prediction,tau,cutoff,*,score_statistics=None):
    p=np.asarray(prediction);tau=np.asarray(tau);alarm=p>cutoff;crossed=alarm.any(axis=1)
    first=np.where(crossed,alarm.argmax(axis=1),-1);null=tau<0;changed=~null
    pre=changed&crossed&(first<tau);detected=changed&crossed&(first>=tau)
    statistics=score_statistics if score_statistics is not None else {'score_q99':float(np.quantile(p,.99)),'mean_max_score':float(p.max(axis=1).mean())}
    return {'threshold':float(cutoff),'null_series':int(null.sum()),'break_series':int(changed.sum()),
        'null_eventual_alarm_rate':float(crossed[null].mean()) if null.any() else None,
        'prebreak_alarm_rate':float(pre[changed].mean()) if changed.any() else None,
        'postbreak_first_detection_rate':float(detected[changed].mean()) if changed.any() else None,
        'conditional_detection_delay_median':float(np.median(first[detected]-tau[detected])) if detected.any() else None,
        **statistics}
