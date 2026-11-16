"""Existing fitted-model sensitivity on identical synthetic seed cohorts."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import sys
import json
import argparse
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next2.counterexamples import policy,generate,predict,source_hash as generator_hash
from src.next2.synthetic_analysis import calibration_map
from src.next2.runtime import prepare,source_hash as runtime_hash
from src.next2.io import model_class
from src.next2.session import checkpoint
from src.utils.artifacts import write_json,sha256,utc_now

OUT=ROOT/'artifacts/next2/finalist_fit_sensitivity'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1)
    args=parser.parse_args()
    if not 0<=args.shard<args.shards:
        raise ValueError('Invalid fixed work shard')
    path=ROOT/'configs/next2/FINALIST_FIT_SENSITIVITY_LOCK.json'
    menu=json.loads(path.read_text())
    lock=policy()
    if menu['counterexample_lock_sha256']!=sha256(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json'):
        raise RuntimeError('Synthetic lock changed')
    new=[r for r in menu['models'] if 'reuse_directory' not in r]
    for index,ref in enumerate(new):
        if index%args.shards!=args.shard:
            continue
        eid,fold=ref['experiment_id'],ref['fold']
        folder=OUT/f'fold_{fold}'/eid
        folder.mkdir(parents=True,exist_ok=True)
        identity={'menu_sha256':sha256(path),'script_sha256':sha256(Path(__file__)),'model_sha256':ref['model_sha256'],
            'runtime_sha256':runtime_hash(),'generator_sha256':generator_hash(),'candidate':eid,'fold':fold}
        if (folder/'RESULTS.json').exists():
            r=json.loads((folder/'RESULTS.json').read_text())
            if r['status']!='PASS' or r['identity']!=identity:
                raise RuntimeError('Fitted-model sensitivity identity changed')
            print('NEXT2 FIT SENSITIVITY RETAINED',eid,fold,flush=True)
            continue
        if sha256(ROOT/ref['model_path'])!=ref['model_sha256']:
            raise RuntimeError('Fitted model bytes changed')
        engine=prepare(model_class(ref['candidate']).load(ROOT/ref['model_path']),resident=False)
        jobs=[('calibration',s,lock['calibration_null_series_per_regime'],lock['online_length']) for s in calibration_map(lock).values()]
        jobs += [('counterexample',s['id'],lock['evaluation_series_per_scenario'],lock['online_length']) for s in lock['scenarios']]
        jobs += [('long_null',s,lock['long_null']['series_per_regime'],max(lock['long_null']['horizons'])) for s in lock['long_null']['regimes']]
        job_receipts={}
        total=time.perf_counter()
        for kind,scenario,n,horizon in jobs:
            out=folder/kind/scenario
            out.mkdir(parents=True,exist_ok=True)
            job={**identity,'kind':kind,'scenario':scenario,'series':n,'horizon':horizon}
            if (out/'RESULTS.json').exists():
                r=json.loads((out/'RESULTS.json').read_text())
                if r['status']!='PASS' or r['identity']!=job or any(sha256(out/name)!=digest for name,digest in r['files_sha256'].items()):
                    raise RuntimeError('Fitted-model sensitivity job changed')
            else:
                write_json(out/'POLICY.json',job)
                paired=kind=='counterexample' and not scenario.startswith('N')
                count=2*n if paired else n
                values=np.lib.format.open_memmap(out/'prediction.npy',mode='w+',dtype=np.float32,shape=(count,horizon))
                taus=np.empty(count,dtype=np.int32)
                tick=time.perf_counter()
                for j in range(count):
                    h,o,tau=generate(scenario,j%n,partition={'calibration':0,'counterexample':1,'long_null':2}[kind],
                        horizon=horizon,control=kind=='calibration' or (paired and j<n),lock=lock)
                    values[j]=predict(engine,h,o)
                    taus[j]=tau
                values.flush()
                np.save(out/'synthetic_tau.npy',taus,allow_pickle=False)
                write_json(out/'RESULTS.json',{'status':'PASS','identity':job,'created_utc':utc_now(),
                    'points':count*horizon,'elapsed_seconds':time.perf_counter()-tick,'seal_rows':0,
                    'artifact_kind':'existing full-CV fitted model','fitted_fold':fold,
                    'files_sha256':{name:sha256(out/name) for name in ('prediction.npy','synthetic_tau.npy')}})
                print('NEXT2 FIT SENSITIVITY',eid,fold,kind,scenario,round(time.perf_counter()-tick,1),flush=True)
                checkpoint('Fit sensitivity '+eid+'/fold '+str(fold)+'/'+kind+'/'+scenario,
                    ['Remaining predeclared fitted-model sensitivity','Compare directions across fits, without new ensemble'])
            job_receipts[(out/'RESULTS.json').relative_to(folder).as_posix()]=sha256(out/'RESULTS.json')
        write_json(folder/'RESULTS.json',{'status':'PASS','identity':identity,'created_utc':utc_now(),
            'job_receipts_sha256':job_receipts,'jobs':len(jobs),'new_fits':0,'elapsed_seconds':time.perf_counter()-total,'seal_rows':0})
        print('NEXT2 FIT SENSITIVITY MODEL COMPLETE',eid,fold,flush=True)


if __name__=='__main__':
    main()
