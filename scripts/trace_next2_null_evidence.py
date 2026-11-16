"""Internal null evidence trajectories with every saved prediction checked."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys
import json
import argparse
import time
import numpy as np
import pandas as pd
from numba import njit
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next2.counterexamples import policy,generate,source_hash as generator_hash
from src.next2.runtime import prepare,source_hash as runtime_hash
from src.next2.io import model_class
from src.next2.session import checkpoint
from src.utils.artifacts import write_json,sha256,utc_now


@njit(cache=False)
def trace(kernel,args,output,points,columns,monotone_columns,horizons):
    snapshots=np.empty((len(horizons),len(columns)),dtype=np.float32)
    maxima=np.empty_like(snapshots);running=np.full(len(columns),-np.inf,dtype=np.float32)
    previous=np.full(len(monotone_columns),-np.inf,dtype=np.float32)
    predictions=np.empty(len(points),dtype=np.float32);violations=0;slot=0
    for t in range(len(points)):
        predictions[t]=kernel(float(points[t]),args)
        for j in range(len(columns)):running[j]=max(running[j],output[columns[j]])
        for j in range(len(monotone_columns)):
            value=output[monotone_columns[j]]
            if value<previous[j]:violations+=1
            previous[j]=value
        if slot<len(horizons) and t+1==horizons[slot]:
            for j in range(len(columns)):snapshots[slot,j]=output[columns[j]];maxima[slot,j]=running[j]
            slot+=1
    return snapshots,maxima,predictions,violations


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=1)
    args=parser.parse_args();assert 0<=args.shard<args.shards
    lock=policy();out=ROOT/'artifacts/next2/null_evidence';out.mkdir(parents=True,exist_ok=True)
    frozen=json.loads((ROOT/'artifacts/next2/initialization/FROZEN_REFERENCES.json').read_text())
    refs=[{'candidate':r['candidate'],'model_path':r['model_path'],'model_sha256':r['model_sha256']} for r in frozen['references'].values()]
    cpath=ROOT/'artifacts/next2/refits/C205_DEPENDENCE/cache_v1'
    refs.append({'candidate':json.loads((cpath/'TRAINING.json').read_text())['candidate'],
        'model_path':str((cpath/'model.joblib').relative_to(ROOT)),'model_sha256':sha256(cpath/'model.joblib')})
    identity={'models':refs,'counterexample_lock_sha256':sha256(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json'),
        'script_sha256':sha256(Path(__file__)),'generator_sha256':generator_hash(),'runtime_sha256':runtime_hash(),
        'protocol_sha256':sha256(ROOT/'docs/NEXT2_NULL_EVIDENCE_PROTOCOL.md'),'new_fits':0,'promotion_authority':False}
    ppath=ROOT/'configs/next2/NULL_EVIDENCE_DIAGNOSTIC_LOCK.json'
    if ppath.exists():assert json.loads(ppath.read_text())['identity']==identity
    else:write_json(ppath,{'created_utc':utc_now(),'identity':identity})
    engines={}
    for ref in refs:
        assert sha256(ROOT/ref['model_path'])==ref['model_sha256']
        engines[ref['candidate']['experiment_id']]=prepare(model_class(ref['candidate']).load(ROOT/ref['model_path']),resident=False)
    horizons=np.asarray([1024,4096,16384],dtype=np.int64)
    for ordinal,bg in enumerate(('N0','N1','N2','N3')):
        if ordinal%args.shards!=args.shard:continue
        for eid,engine in engines.items():
            folder=out/bg/eid;folder.mkdir(parents=True,exist_ok=True)
            if (folder/'RESULTS.json').exists():
                r=json.loads((folder/'RESULTS.json').read_text());assert r['status']=='PASS' and r['identity']==identity
                for f,digest in r['files_sha256'].items():assert sha256(folder/f)==digest
                continue
            source=ROOT/('artifacts/next2/finalist_counterexamples' if eid=='C205_DEPENDENCE' else 'artifacts/next2/counterexamples')/'long_null'/bg
            receipt=json.loads((source/'RESULTS.json').read_text());path=source/(eid+'.npy')
            assert receipt['status']=='PASS' and sha256(path)==receipt['files_sha256'][path.name]
            expected=np.load(path,mmap_mode='r',allow_pickle=False);assert expected.shape==(512,16384)
            names=engine.model.names
            columns=np.asarray([j for j,name in enumerate(names) if not name.startswith('arch8__') or 'cusum' in name],dtype=np.int64)
            monotone=np.asarray([j for j,name in enumerate(names) if name.endswith('_memory_max')],dtype=np.int64)
            current=np.empty((512,3,len(columns)),dtype=np.float32);maxima=np.empty_like(current);score_max=np.empty((512,3),dtype=np.float32)
            tick=time.perf_counter();violations=0
            for index in range(512):
                h,o,tau=generate(bg,index,partition=2,horizon=16384,control=False,lock=lock);assert tau==-1
                state=engine.new_state(h)
                a,b,p,bad=trace(state.kernel,state.call_args,state.output,o,columns,monotone,horizons)
                np.testing.assert_array_equal(p,expected[index]);current[index]=a;maxima[index]=b;violations+=bad
                score_max[index]=[p[:horizon].max() for horizon in horizons]
            assert violations==0
            np.savez_compressed(folder/'SNAPSHOTS.npz',current=current,path_max=maxima,score_path_max=score_max)
            selected_names=[names[j] for j in columns];write_json(folder/'FEATURE_NAMES.json',selected_names)
            rows=[]
            for k,horizon in enumerate(horizons):
                for j,name in enumerate(selected_names):
                    for kind,values in [('current',current),('path_max',maxima)]:
                        delta=values[:,k,j]-values[:,0,j]
                        row={'candidate':eid,'background':bg,'horizon':int(horizon),'feature':name,'quantity':kind,'series':512,
                            'mean':float(values[:,k,j].mean(dtype=np.float64)),
                            'paired_mean_change_from_1024':float(delta.mean(dtype=np.float64)),
                            'paired_median_change_from_1024':float(np.median(delta)),
                            'artifact_kind':'frozen all-DEV model; original partition-2 null cohort'}
                        row.update({f'quantile_{q}':float(v) for q,v in zip((.5,.9,.95,.99),np.quantile(values[:,k,j],[.5,.9,.95,.99]))})
                        rows.append(row)
            pd.DataFrame(rows).to_csv(folder/'FEATURE_GROWTH.csv',index=False)
            write_json(folder/'RESULTS.json',{'status':'PASS','identity':identity,'created_utc':utc_now(),'candidate':eid,'background':bg,
                'series':512,'points':512*16384,'every_saved_prediction_exact':True,'cumulative_max_memory_decreases':violations,
                'cumulative_max_features_checked':[names[j] for j in monotone],'source_prediction_sha256':sha256(path),
                'elapsed_seconds':time.perf_counter()-tick,'new_fits':0,'seal_rows':0,
                'files_sha256':{name:sha256(folder/name) for name in ('SNAPSHOTS.npz','FEATURE_NAMES.json','FEATURE_GROWTH.csv')}})
            print('NEXT2 NULL EVIDENCE',bg,eid,round(time.perf_counter()-tick,1),flush=True)
            checkpoint('Null internal evidence '+bg+'/'+eid,['Connect raw evidence growth and bounded score alarms','Preserve synthetic-only interpretation'])


if __name__=='__main__':main()
