"""Fixed diagnostic cohorts, calibrated alarms and matched prediction sources."""
import math
from pathlib import Path
import numpy as np
from src.next4.artifacts import ROOT,BASE,read_json,digest
from src.next2.counterexamples import generate,policy as original_policy,predict
from src.next2.cross_background_scale import series as crossed_series
from src.next2.blend import combine,FixedBlendBundle,source_hash as blend_hash


def threshold(calibration_maxima,alpha):
    values=np.asarray(calibration_maxima,dtype=np.float64)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all() or not 0<alpha<1:
        raise ValueError('Finite nonempty calibration maxima and alpha in (0,1) required')
    rank=math.ceil((len(values)+1)*(1-alpha))
    return float(np.sort(values)[rank-1]) if rank<=len(values) else 1.


def alarm_summary(prediction,tau,cutoff):
    p=np.asarray(prediction);tau=np.asarray(tau)
    if p.ndim!=2 or tau.shape!=(len(p),) or not np.isfinite(p).all():raise ValueError('Aligned finite diagnostic rows required')
    alarm=p>cutoff;crossed=alarm.any(axis=1);first=np.where(crossed,alarm.argmax(axis=1),-1)
    null=tau<0;changed=~null;pre=changed&crossed&(first<tau);detected=changed&crossed&(first>=tau)
    return {'threshold':cutoff,'null_series':int(null.sum()),'break_series':int(changed.sum()),
        'null_eventual_alarm_rate':float(crossed[null].mean()) if null.any() else None,
        'prebreak_alarm_rate':float(pre[changed].mean()) if changed.any() else None,
        'postbreak_first_detection_rate':float(detected[changed].mean()) if changed.any() else None,
        'conditional_detection_delay_median':float(np.median(first[detected]-tau[detected])) if detected.any() else None,
        'score_q99':float(np.quantile(p,.99)),'mean_max_score':float(p.max(axis=1).mean())}


def background_mapping(lock):
    result={}
    for spec in lock['scenarios']:
        result.setdefault((spec['phi_before'],spec['arch_before'],spec['t_df_before']),spec['id'])
    return result


def scenario_background(scenario,lock):
    if scenario.startswith('CB_N3_'):scenario='N3'
    if scenario=='OB16':scenario='O1'
    spec=next(s for s in lock['scenarios'] if s['id']==scenario)
    return (spec['phi_before'],spec['arch_before'],spec['t_df_before'])


def cohort_series(kind,scenario,index,horizon,control,lock):
    if kind not in ('calibration','counterexample','long_null'):raise ValueError('Fixed diagnostic partition required')
    partition={'calibration':0,'counterexample':1,'long_null':2}[kind]
    if scenario.startswith('CB_N3_'):
        if kind!='counterexample':raise ValueError('Crossed factors are diagnostic counterexamples only')
        factor={'CB_N3_070':.7,'CB_N3_050':.5}[scenario]
        return crossed_series(lock,'N3',factor,index,control=control,horizon=horizon)
    if scenario=='OB16':
        if kind!='counterexample':raise ValueError('Burst is a paired counterexample only')
        h,o,tau=generate('O1',index,partition=1,horizon=horizon,control=True,lock=lock)
        if not control:
            start=int(lock['tau_cycle'][index%len(lock['tau_cycle'])]);stop=min(start+16,len(o))
            o[start:stop]+=np.asarray([8.*(1 if (index+j)%2==0 else -1) for j in range(stop-start)],dtype=np.float32)
        return h,o,-1
    return generate(scenario,index,partition=partition,horizon=horizon,control=control,lock=lock)


def paired_seed_counts(n,replicates,seed):
    if n<2 or replicates<1:raise ValueError('Nontrivial paired seed bootstrap required')
    rng=np.random.default_rng(seed)
    counts=np.vstack([np.ones((1,n),dtype=np.int32),
        np.asarray([np.bincount(row,minlength=n) for row in rng.integers(0,n,(replicates,n))],dtype=np.int32)])
    return counts


def load_reference(training_scope):
    from src.next2.shared_blend_runtime import PreparedSharedBlendPredictor
    if training_scope=='all_dev':
        from src.next2.shared_inference import load_artifact
        directory=ROOT/'artifacts/next2/canonical_deployment/PERFORMANCE_shared_v1'
        model,descriptor=load_artifact(directory)
        proof={'training_scope':'ALL_DEV_8000','canonical_model_sha256':digest(directory/'model.sbnext2'),
            'descriptor_sha256':digest(directory/'model.json')}
    elif training_scope in ('full_fold0','screen_fold0'):
        from src.next.data import prepare_dev
        from src.next3.frozen import fold_artifact,J102,C205
        data=prepare_dev(ROOT);scope=training_scope.split('_')[0]
        j,_,jr,jproof=fold_artifact(data,J102,0,scope);c,_,cr,cproof=fold_artifact(data,C205,0,scope)
        np.testing.assert_array_equal(jr,cr)
        model=FixedBlendBundle(j,c,.8,implementation_sha256=blend_hash());model.validate()
        proof={'training_scope':training_scope,'original_fold':0,'J102_files_sha256':jproof,'C205_files_sha256':cproof}
        data.guard.check_state()
    else:raise ValueError('Explicit matched training scope required')
    prepared=PreparedSharedBlendPredictor(model,resident=False)
    return prepared,proof


def load_candidate(nomination):
    from src.next4.model import FeatureModel
    from src.next4.state_readout import StateReadout
    path=ROOT/nomination['model_path']
    if digest(path)!=nomination['model_sha256']:raise RuntimeError('Nominated diagnostic model changed')
    cls=StateReadout if nomination['family']=='H' else FeatureModel
    model=cls.load(path)
    return model


def candidate_prediction(model,nomination,h,o,reference,complement):
    state=model.make_state(h);x=state.replay(o);expert=np.asarray(model.predict(x),dtype=np.float32)
    if nomination['family']=='B':return combine(expert,complement,.8)
    return combine(reference,expert,.9)


def direct_candidate_prediction(model,nomination,h,o,reference,complement):
    state=model.make_state(h)
    expert=np.asarray([model.predict_one(state.update(float(point))) for point in o],dtype=np.float32)
    return combine(expert,complement,.8) if nomination['family']=='B' else combine(reference,expert,.9)


def sources():
    names=['src/next4/synthetic.py','scripts/stress_next4_candidate.py','src/next2/counterexamples.py',
        'src/next2/cross_background_scale.py','src/next2/blend.py','src/next2/shared_blend_runtime.py',
        'src/next2/dependence_runtime.py','src/next3/frozen.py','src/next/bootstrap.py',
        'src/validation/group_bootstrap.py','src/scoring/ts_auc.py']
    return {name:digest(ROOT/name) for name in names}
