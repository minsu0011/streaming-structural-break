"""Prospective long-late-break diagnostic using unchanged recurrence and frozen models."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import sys
import argparse
import json
import time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next2.counterexamples import policy,generate,predict,source_hash as generator_hash
from src.next2.cross_background_scale import crossed_lock
from src.next2.synthetic_analysis import score,threshold,alarm_summary,age_normalize
from src.next2.runtime import prepare,source_hash as runtime_hash
from src.next2.io import model_class
from src.next2.blend import combine
from src.next2.session import checkpoint
from src.utils.artifacts import write_json,sha256,utc_now

J='J102_SERIAL_VARIANCE_512';C='C205_DEPENDENCE';B='BL2_C205_DEPENDENCE_80'
OUT=ROOT/'artifacts/next2/late_alarm_diagnostic'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=1)
    args=parser.parse_args();assert 0<=args.shard<args.shards
    OUT.mkdir(parents=True,exist_ok=True)
    original=policy();frozen=json.loads((ROOT/'artifacts/next2/initialization/FROZEN_REFERENCES.json').read_text())
    refs=[{'candidate':r['candidate'],'model_path':r['model_path'],'model_sha256':r['model_sha256']} for r in frozen['references'].values()]
    folder=ROOT/'artifacts/next2/refits/C205_DEPENDENCE/cache_v1'
    refs.append({'candidate':json.loads((folder/'TRAINING.json').read_text())['candidate'],
        'model_path':(folder/'model.joblib').relative_to(ROOT).as_posix(),'model_sha256':sha256(folder/'model.joblib')})
    cells=[{'background':bg,'onset':onset} for bg in ('N0','N2') for onset in (4096,49152)]
    identity={'models':refs,'cells':cells,'partition':4,'seed_pairs_per_cell':128,'horizon':65536,'variance_factor':2.,'betas':[0.,.5,1.],
        'counterexample_lock_sha256':sha256(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json'),'generator_sha256':generator_hash(),
        'crossing_source_sha256':sha256(ROOT/'src/next2/cross_background_scale.py'),'runtime_sha256':runtime_hash(),
        'script_sha256':sha256(Path(__file__)),'protocol_sha256':sha256(ROOT/'docs/NEXT2_LATE_ALARM_PROTOCOL.md'),
        'new_fits':0,'promotion_authority':False}
    nomination=ROOT/'configs/next2/LATE_ALARM_DIAGNOSTIC_LOCK.json'
    if nomination.exists():assert json.loads(nomination.read_text())['identity']==identity
    else:write_json(nomination,{'created_utc':utc_now(),'identity':identity})
    engines={}
    for ref in refs:
        assert sha256(ROOT/ref['model_path'])==ref['model_sha256']
        engines[ref['candidate']['experiment_id']]=prepare(model_class(ref['candidate']).load(ROOT/ref['model_path']),resident=False)
    all_models=list(engines)+[B];n=128;horizon=65536
    for ordinal,cell in enumerate(cells):
        if ordinal%args.shards!=args.shard:continue
        bg,onset=cell['background'],cell['onset'];label=bg+'_ONSET_'+str(onset);out=OUT/label;out.mkdir(parents=True,exist_ok=True)
        if (out/'RESULTS.json').exists():
            r=json.loads((out/'RESULTS.json').read_text());assert r['identity']==identity
            for name,digest in r['files_sha256'].items():assert sha256(out/name)==digest
        else:
            write_json(out/'POLICY.json',{'identity':identity,'cell':cell})
            values={eid:np.lib.format.open_memmap(out/(eid+'.npy'),mode='w+',dtype=np.float32,shape=(2*n,horizon)) for eid in all_models}
            tau=np.full(2*n,-1,dtype=np.int32);lock=crossed_lock(original,2.);lock['tau_cycle']=[onset]
            tick=time.perf_counter()
            for index in range(n):
                h0,o0,t0=generate('CB_'+bg,index,partition=4,control=True,horizon=horizon,lock=lock)
                h1,o1,t1=generate('CB_'+bg,index,partition=4,control=False,horizon=horizon,lock=lock)
                assert t0==-1 and t1==onset;np.testing.assert_array_equal(h0,h1);np.testing.assert_array_equal(o0[:onset],o1[:onset])
                tau[n+index]=t1
                for eid,engine in engines.items():
                    values[eid][index]=predict(engine,h0,o0);values[eid][n+index]=predict(engine,h1,o1)
                for row in (index,n+index):values[B][row]=combine(values[J][row],values[C][row],.8)
                if (index+1)%16==0:
                    print('NEXT2 LATE ALARM',label,index+1,round(time.perf_counter()-tick,1),flush=True)
                    checkpoint('Late alarm '+label+'/'+str(index+1),['Finish locked long-late cohorts','Measure detection versus null tradeoff'])
            for value in values.values():value.flush()
            np.save(out/'synthetic_tau.npy',tau,allow_pickle=False)
            write_json(out/'RESULTS.json',{'status':'PASS','identity':identity,'created_utc':utc_now(),'cell':cell,
                'elapsed_seconds':time.perf_counter()-tick,'new_fits':0,'seal_rows':0,
                'files_sha256':{name:sha256(out/name) for name in [eid+'.npy' for eid in all_models]+['synthetic_tau.npy']}})
        if (out/'ANALYSIS.json').exists():
            print('NEXT2 LATE ANALYSIS RETAINED',label,flush=True);continue
        tau=np.load(out/'synthetic_tau.npy',allow_pickle=False);rows=[]
        for eid in all_models:
            p=np.load(out/(eid+'.npy'),mmap_mode='r',allow_pickle=False)
            # Only the common post-change interval contains positive/negative pairs.
            # Removing its all-negative prefix leaves the pair-weighted AUC unchanged.
            auc=score(p[:,onset:],np.r_[np.full(n,-1),np.zeros(n,dtype=np.int32)])
            source_root=ROOT/('artifacts/next2/finalist_counterexamples' if eid in (C,B) else 'artifacts/next2/counterexamples')
            calibration=source_root/'calibration'/bg
            receipt=json.loads((calibration/'RESULTS.json').read_text());assert sha256(calibration/(eid+'.npy'))==receipt['files_sha256'][eid+'.npy']
            cp=np.load(calibration/(eid+'.npy'),mmap_mode='r',allow_pickle=False)
            for beta in identity['betas']:
                maxima=age_normalize(cp,np.arange(cp.shape[1]),beta).max(axis=1)
                q=age_normalize(p,np.arange(p.shape[1]),beta)
                for alpha in original['long_null']['threshold_alphas']:
                    level=threshold(maxima,alpha);summary,_,_=alarm_summary(q,tau,level)
                    rows.append({'candidate':eid,'background':bg,'onset':onset,'horizon':horizon,'variance_factor':2.,
                        'beta':beta,'alpha':alpha,'threshold':level,'raw_ts_auc':auc,**summary,
                        'artifact_kind':'frozen all-DEV model; prospective long-late supplement','promotion_authority':False})
            print('NEXT2 LATE ANALYSIS',label,eid,auc,flush=True)
        pd.DataFrame(rows).to_csv(out/'ANALYSIS.csv',index=False)
        write_json(out/'ANALYSIS.json',{'status':'PASS','identity':identity,'raw_result_sha256':sha256(out/'RESULTS.json'),
            'table_sha256':sha256(out/'ANALYSIS.csv'),'ts_auc_note':'All-negative pre-onset times have no within-time positive/negative pairs and are omitted only from score calculation.'})
        checkpoint('Completed long-late analysis '+label,['Compare all four cells','Keep detection costs explicit'])


if __name__=='__main__':
    main()
