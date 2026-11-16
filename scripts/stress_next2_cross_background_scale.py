"""Prospective complete 4-background x 4-variance supplemental diagnostic, zero model fits."""
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
from src.next2.counterexamples import policy,predict,generate,source_hash as generator_hash
from src.next2.cross_background_scale import crossed_lock,FACTORS,BACKGROUNDS
from src.next2.runtime import prepare,source_hash as runtime_hash
from src.next2.io import model_class
from src.next2.blend import combine
from src.next2.synthetic_analysis import score,threshold,alarm_summary
from src.next2.session import checkpoint
from src.utils.artifacts import write_json,sha256,utc_now

J='J102_SERIAL_VARIANCE_512'
C='C205_DEPENDENCE'
B='BL2_C205_DEPENDENCE_80'


def main():
    outroot=ROOT/'artifacts/next2/cross_background_scale';outroot.mkdir(parents=True,exist_ok=True)
    original=policy()
    frozen=json.loads((ROOT/'artifacts/next2/initialization/FROZEN_REFERENCES.json').read_text())
    refs=[{'candidate':r['candidate'],'model_path':r['model_path'],'model_sha256':r['model_sha256']} for r in frozen['references'].values()]
    folder=ROOT/'artifacts/next2/refits/C205_DEPENDENCE/cache_v1'
    refs.append({'candidate':json.loads((folder/'TRAINING.json').read_text())['candidate'],
        'model_path':(folder/'model.joblib').relative_to(ROOT).as_posix(),'model_sha256':sha256(folder/'model.joblib')})
    identity={'models':refs,'factors':list(FACTORS),'backgrounds':list(BACKGROUNDS),'partition':3,
        'counterexample_lock_sha256':sha256(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json'),'generator_sha256':generator_hash(),
        'crossing_source_sha256':sha256(ROOT/'src/next2/cross_background_scale.py'),'runtime_sha256':runtime_hash(),
        'script_sha256':sha256(Path(__file__)),'protocol_sha256':sha256(ROOT/'docs/NEXT2_CROSS_BACKGROUND_SCALE_PROTOCOL.md'),
        'new_fits':0,'primary_suite_unchanged':True,'promotion_authority':False}
    nomination=ROOT/'configs/next2/CROSS_BACKGROUND_SCALE_LOCK.json'
    if nomination.exists():
        assert json.loads(nomination.read_text())['identity']==identity
    else:
        write_json(nomination,{'created_utc':utc_now(),'identity':identity})
    engines={}
    for ref in refs:
        assert sha256(ROOT/ref['model_path'])==ref['model_sha256']
        engines[ref['candidate']['experiment_id']]=prepare(model_class(ref['candidate']).load(ROOT/ref['model_path']),resident=False)
    all_models=list(engines)+[B]
    n=original['evaluation_series_per_scenario'];horizon=original['online_length']
    rows=[];receipts={}
    for background in BACKGROUNDS:
        controls=outroot/'control'/background;controls.mkdir(parents=True,exist_ok=True)
        if (controls/'RESULTS.json').exists():
            r=json.loads((controls/'RESULTS.json').read_text());assert r['identity']==identity
            for name,digest in r['files_sha256'].items():assert sha256(controls/name)==digest
        else:
            values={eid:np.lib.format.open_memmap(controls/(eid+'.npy'),mode='w+',dtype=np.float32,shape=(n,horizon)) for eid in all_models}
            lock=crossed_lock(original,FACTORS[0]);tick=time.perf_counter()
            for index in range(n):
                h,o,tau=generate('CB_'+background,index,partition=3,control=True,lock=lock)
                assert tau==-1
                for eid,engine in engines.items():values[eid][index]=predict(engine,h,o)
                values[B][index]=combine(values[J][index],values[C][index],.8)
            for value in values.values():value.flush()
            write_json(controls/'RESULTS.json',{'status':'PASS','identity':identity,'created_utc':utc_now(),
                'files_sha256':{eid+'.npy':sha256(controls/(eid+'.npy')) for eid in all_models}})
            print('NEXT2 CROSS BACKGROUND CONTROL',background,round(time.perf_counter()-tick,1),flush=True)
        for factor in FACTORS:
            label=background+'_VAR_'+str(factor).replace('.','p');out=outroot/'evaluation'/label;out.mkdir(parents=True,exist_ok=True)
            if (out/'RESULTS.json').exists():
                r=json.loads((out/'RESULTS.json').read_text());assert r['identity']==identity
                for name,digest in r['files_sha256'].items():assert sha256(out/name)==digest
            else:
                values={eid:np.lib.format.open_memmap(out/(eid+'.npy'),mode='w+',dtype=np.float32,shape=(2*n,horizon)) for eid in all_models}
                for eid in all_models:values[eid][:n]=np.load(controls/(eid+'.npy'),mmap_mode='r',allow_pickle=False)
                tau=np.full(2*n,-1,dtype=np.int32);lock=crossed_lock(original,factor);tick=time.perf_counter()
                for index in range(n):
                    h,o,t=generate('CB_'+background,index,partition=3,control=False,lock=lock);tau[n+index]=t
                    for eid,engine in engines.items():values[eid][n+index]=predict(engine,h,o)
                    values[B][n+index]=combine(values[J][n+index],values[C][n+index],.8)
                for value in values.values():value.flush()
                np.save(out/'synthetic_tau.npy',tau,allow_pickle=False)
                write_json(out/'RESULTS.json',{'status':'PASS','identity':identity,'created_utc':utc_now(),'background':background,
                    'variance_factor':factor,'series':2*n,'source_control_sha256':sha256(controls/'RESULTS.json'),
                    'files_sha256':{name:sha256(out/name) for name in [eid+'.npy' for eid in all_models]+['synthetic_tau.npy']},
                    'elapsed_seconds':time.perf_counter()-tick,'new_fits':0,'seal_rows':0})
                print('NEXT2 CROSS BACKGROUND',label,round(time.perf_counter()-tick,1),flush=True)
                checkpoint('Fixed cross-background scale '+label,['Complete all 16 predeclared cells','Retain original suite results'])
            receipts[(out/'RESULTS.json').relative_to(ROOT).as_posix()]=sha256(out/'RESULTS.json')
            tau=np.load(out/'synthetic_tau.npy',allow_pickle=False)
            for eid in all_models:
                p=np.load(out/(eid+'.npy'),mmap_mode='r',allow_pickle=False)
                source_root=ROOT/('artifacts/next2/finalist_counterexamples' if eid in (C,B) else 'artifacts/next2/counterexamples')
                calibration=source_root/'calibration'/background
                r=json.loads((calibration/'RESULTS.json').read_text());assert sha256(calibration/(eid+'.npy'))==r['files_sha256'][eid+'.npy']
                maxima=np.load(calibration/(eid+'.npy'),mmap_mode='r',allow_pickle=False).max(axis=1)
                auc=score(p,tau)
                for alpha in original['long_null']['threshold_alphas']:
                    level=threshold(maxima,alpha);summary,_,_=alarm_summary(p,tau,level)
                    rows.append({'candidate':eid,'scenario':label,'background':background,'variance_factor':factor,
                        'alpha':alpha,'ts_auc':auc,'threshold':level,**summary,'artifact_kind':'frozen all-DEV model; prospective cross-background supplement',
                        'promotion_authority':False})
            pd.DataFrame(rows).to_csv(outroot/'RESULTS.csv',index=False)
    write_json(outroot/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'cells':16,'models':len(all_models),
        'nomination_sha256':sha256(nomination),'evaluation_receipts_sha256':receipts,'table_sha256':sha256(outroot/'RESULTS.csv'),
        'original_suite_unchanged':True,'new_fits':0,'promotion_authority':False})


if __name__=='__main__':
    main()
