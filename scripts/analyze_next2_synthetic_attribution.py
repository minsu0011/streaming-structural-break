"""All frozen synthetic seeds: exact feature snapshots and additive tree explanations."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import sys
import json
import time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next2.counterexamples import policy,generate,source_hash as generator_hash
from src.next2.runtime import prepare,source_hash as runtime_hash
from src.next2.io import model_class
from src.next2.session import checkpoint
from src.utils.artifacts import write_json,sha256,utc_now


def group(name):
    if name.startswith('arch8__'):
        return 'LEGACY_ABCD'
    channel='ENERGY' if '__raw_energy_' in name else 'FORECAST' if '__log_variance_forecast_' in name else 'DEPENDENCE'
    statistic='BAYES' if '_bayes' in name else 'MEMORY' if '_memory_' in name else 'GLR' if '_glr_' in name else 'LOCAL'
    return channel+'_'+statistic


def main():
    outroot=ROOT/'artifacts/next2/synthetic_attribution'
    outroot.mkdir(parents=True,exist_ok=True)
    frozen=json.loads((ROOT/'artifacts/next2/initialization/FROZEN_REFERENCES.json').read_text())
    models=[{'candidate':r['candidate'],'model_path':r['model_path'],'model_sha256':r['model_sha256']} for r in frozen['references'].values()]
    cdir=ROOT/'artifacts/next2/refits/C205_DEPENDENCE/cache_v1'
    c=json.loads((cdir/'TRAINING.json').read_text())['candidate']
    models.append({'candidate':c,'model_path':(cdir/'model.joblib').relative_to(ROOT).as_posix(),'model_sha256':sha256(cdir/'model.joblib')})
    identity={'models':models,'script_sha256':sha256(Path(__file__)),'runtime_sha256':runtime_hash(),
        'generator_sha256':generator_hash(),'suite_sha256':sha256(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json'),
        'protocol_sha256':sha256(ROOT/'docs/NEXT2_SYNTHETIC_ATTRIBUTION_PROTOCOL.md'),'new_fits':0}
    nomination=outroot/'POLICY.json'
    if nomination.exists():
        assert json.loads(nomination.read_text())==identity
    else:
        write_json(nomination,identity)
    lock=policy();n=lock['evaluation_series_per_scenario']
    all_summaries=[];all_deltas=[];receipts={}
    for ref in models:
        eid=ref['candidate']['experiment_id']
        assert sha256(ROOT/ref['model_path'])==ref['model_sha256']
        model=model_class(ref['candidate']).load(ROOT/ref['model_path'])
        engine=prepare(model)
        names=np.asarray(model.names)[model.columns]
        groups=sorted(set(group(name) for name in names))
        group_indices={g:[j for j,name in enumerate(names) if group(name)==g] for g in groups}
        source_root=ROOT/('artifacts/next2/finalist_counterexamples' if eid=='C205_DEPENDENCE' else 'artifacts/next2/counterexamples')
        for spec in lock['scenarios']:
            scenario=spec['id'];out=outroot/eid/scenario;out.mkdir(parents=True,exist_ok=True)
            done=out/'RESULTS.json'
            if done.exists():
                r=json.loads(done.read_text());assert r['status']=='PASS' and r['policy_sha256']==sha256(nomination)
                for name,digest in r['files_sha256'].items():
                    assert sha256(out/name)==digest
            else:
                source=source_root/'counterexample'/scenario
                receipt=json.loads((source/'RESULTS.json').read_text())
                for name in (eid+'.npy','synthetic_tau.npy'):
                    assert sha256(source/name)==receipt['files_sha256'][name]
                expected=np.load(source/(eid+'.npy'),mmap_mode='r',allow_pickle=False)
                tau_array=np.load(source/'synthetic_tau.npy',allow_pickle=False)
                paired=not scenario.startswith('N');count=2*n if paired else n
                assert expected.shape==(count,lock['online_length'])
                features=[];coordinates=[];predictions=[]
                tick=time.perf_counter()
                for j in range(count):
                    index=j%n;control=paired and j<n
                    h,o,tau=generate(scenario,index,partition=1,horizon=lock['online_length'],control=control,lock=lock)
                    assert tau==tau_array[j]
                    scheduled=lock['tau_cycle'][index%len(lock['tau_cycle'])]
                    slots={'t0':0,'t31':31,'pre1':scheduled-1,'post0':scheduled,'post31':scheduled+31,
                           'post127':scheduled+127,'post511':scheduled+511,'t1023':1023}
                    inverse={t:[name for name,value in slots.items() if value==t] for t in set(slots.values())}
                    state=engine.new_state(h)
                    for t,value in enumerate(o):
                        prediction=state.predict_one(value)
                        assert np.float32(prediction)==expected[j,t]
                        if t in inverse:
                            for slot in inverse[t]:
                                features.append(state.output.copy())
                                coordinates.append({'candidate':eid,'scenario':scenario,'seed_index':index,
                                    'arm':'control' if control else 'intervention' if paired else 'null',
                                    'slot':slot,'time_online':t,'scheduled_change':scheduled,'synthetic_tau':tau})
                                predictions.append(prediction)
                x=np.asarray(features,dtype=np.float32)
                selected=x[:,model.columns]
                contributions=model.estimator.booster_.predict(selected,pred_contrib=True,num_threads=1)
                margin=model.estimator.booster_.predict(selected,raw_score=True,num_threads=1)
                np.testing.assert_allclose(contributions.sum(axis=1),margin,rtol=0,atol=1e-10)
                np.testing.assert_allclose(1./(1.+np.exp(-margin)),predictions,rtol=0,atol=1e-7)
                frame=pd.DataFrame(coordinates);frame['prediction']=predictions;frame['raw_margin']=margin
                frame['BASE_VALUE']=contributions[:,-1]
                for g,indices in group_indices.items():
                    frame[g]=contributions[:,indices].sum(axis=1)
                np.testing.assert_allclose(frame[groups+['BASE_VALUE']].sum(axis=1),margin,rtol=0,atol=1e-10)
                frame.to_csv(out/'SNAPSHOTS.csv',index=False)
                np.save(out/'FEATURE_SNAPSHOTS.npy',x,allow_pickle=False)
                summary=frame.groupby(['candidate','scenario','arm','slot'])[groups+['BASE_VALUE','prediction','raw_margin']].mean().reset_index()
                summary.to_csv(out/'GROUP_MEANS.csv',index=False)
                deltas=[]
                if paired:
                    controls=frame[frame.arm=='control'].set_index(['seed_index','slot'])
                    interventions=frame[frame.arm=='intervention'].set_index(['seed_index','slot'])
                    delta=interventions[groups+['prediction','raw_margin']]-controls[groups+['prediction','raw_margin']]
                    for slot,part in delta.groupby(level='slot'):
                        for metric in part:
                            values=part[metric].to_numpy()
                            deltas.append({'candidate':eid,'scenario':scenario,'slot':slot,'metric':metric,'seed_pairs':len(values),
                                'paired_mean_delta':float(values.mean()),'paired_median_delta':float(np.median(values)),
                                'paired_q025':float(np.quantile(values,.025)),'paired_q975':float(np.quantile(values,.975)),
                                'interval_type':'empirical pair spread, not confidence interval'})
                pd.DataFrame(deltas,columns=['candidate','scenario','slot','metric','seed_pairs','paired_mean_delta','paired_median_delta','paired_q025','paired_q975','interval_type']).to_csv(out/'PAIRED_GROUP_DELTAS.csv',index=False)
                write_json(done,{'status':'PASS','policy_sha256':sha256(nomination),'created_utc':utc_now(),'model':eid,
                    'scenario':scenario,'synthetic_series':count,'all_online_predictions_checked':int(expected.size),
                    'snapshots':len(frame),'source_receipt_sha256':sha256(source/'RESULTS.json'),
                    'additivity_max_abs_diff':float(np.max(np.abs(contributions.sum(1)-margin))),
                    'feature_names':list(model.names),'groups':group_indices,'elapsed_seconds':time.perf_counter()-tick,
                    'new_fits':0,'causal_claim':False,'files_sha256':{name:sha256(out/name) for name in ('SNAPSHOTS.csv','FEATURE_SNAPSHOTS.npy','GROUP_MEANS.csv','PAIRED_GROUP_DELTAS.csv')}})
                print('NEXT2 SYNTHETIC ATTRIBUTION',eid,scenario,round(time.perf_counter()-tick,1),flush=True)
                checkpoint('Synthetic tree attribution '+eid+'/'+scenario,['Continue all fixed cases','Contrast scale-down and dependence contributions'])
            receipts[done.relative_to(ROOT).as_posix()]=sha256(done)
            all_summaries.append(pd.read_csv(out/'GROUP_MEANS.csv'))
            all_deltas.append(pd.read_csv(out/'PAIRED_GROUP_DELTAS.csv'))
        pd.concat(all_summaries,ignore_index=True).to_csv(outroot/'GROUP_MEANS.csv',index=False)
        pd.concat(all_deltas,ignore_index=True).to_csv(outroot/'PAIRED_GROUP_DELTAS.csv',index=False)
    write_json(outroot/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'models':len(models),'scenarios_per_model':18,
        'receipts_sha256':receipts,'new_fits':0,'policy_sha256':sha256(nomination),'interpretation':'Fixed predictive decomposition, not causal ablation.'})


if __name__=='__main__':
    main()
