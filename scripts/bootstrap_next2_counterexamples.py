"""Paired synthetic uncertainty: each control/changed seed pair is one unit."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import sys
import json
import time
import numpy as np
import pandas as pd
from numba import set_num_threads
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next2.counterexamples import policy
from src.next.bootstrap import PreparedAUC
from src.validation.group_bootstrap import draw_group_multiplicities
from src.scoring.ts_auc import ts_auc
from src.next2.session import checkpoint
from src.utils.artifacts import write_json,sha256,utc_now

J='J102_SERIAL_VARIANCE_512'
MODELS=(J,'Y001_LONG_VARIANCE_ARCH1','B104_LONG_VARIANCE_NO_GLR_MEMORY','C205_DEPENDENCE','BL2_C205_DEPENDENCE_80')


def paired_group_codes(pair_count,horizon):
    return np.tile(np.repeat(np.arange(pair_count,dtype=np.int32),horizon),2)


def main():
    set_num_threads(2)
    lock=policy()
    out=ROOT/'artifacts/next2/counterexample_bootstrap'
    out.mkdir(parents=True,exist_ok=True)
    identity={'script_sha256':sha256(Path(__file__)),'bootstrap_source_sha256':sha256(ROOT/'src/next/bootstrap.py'),
        'counterexample_lock_sha256':sha256(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json'),'replicates':2000,'seed_base':2026090927,
        'resampling_unit':'Each common seed control/changed pair, keeping both whole timelines together',
        'models':list(MODELS),'uncertainty':'Conditional on frozen fitted models and specified DGP; no synthetic model selection'}
    if (out/'POLICY.json').exists():
        if json.loads((out/'POLICY.json').read_text())!=identity:
            raise RuntimeError('Paired synthetic bootstrap policy changed')
    else:
        write_json(out/'POLICY.json',identity)
    rows=[]
    sources={}
    started=time.perf_counter()
    for ordinal,spec in enumerate(lock['scenarios']):
        scenario=spec['id']
        if scenario.startswith('N') or scenario=='O1':
            continue
        folder=out/scenario
        folder.mkdir(exist_ok=True)
        reference=ROOT/'artifacts/next2/counterexamples/counterexample'/scenario
        tau=np.load(reference/'synthetic_tau.npy',allow_pickle=False)
        n=lock['evaluation_series_per_scenario']
        h=lock['online_length']
        if tau.shape!=(2*n,) or np.any(tau[:n]>=0) or np.any(tau[n:]<0):
            raise RuntimeError('Expected paired control and changed cohort ordering')
        t=np.broadcast_to(np.arange(h,dtype=np.int32),(2*n,h))
        y=((t>=tau[:,None])&(tau[:,None]>=0)).astype(np.uint8)
        codes=paired_group_codes(n,h)
        _,counts=draw_group_multiplicities(np.arange(n),2000,2026090927+ordinal)
        matrices={}
        for eid in MODELS:
            root=ROOT/('artifacts/next2/counterexamples' if eid in MODELS[:3] else 'artifacts/next2/finalist_counterexamples')
            source=root/'counterexample'/scenario
            r=json.loads((source/'RESULTS.json').read_text())
            path=source/(eid+'.npy')
            if r['status']!='PASS' or sha256(path)!=r['files_sha256'][path.name]:
                raise RuntimeError('Synthetic bootstrap source changed')
            np.testing.assert_array_equal(np.load(source/'synthetic_tau.npy',allow_pickle=False),tau)
            sources[path.relative_to(ROOT).as_posix()]=r['files_sha256'][path.name]
            target=folder/(eid+'.npy')
            receipt=folder/(eid+'.json')
            if target.exists() and receipt.exists():
                saved=json.loads(receipt.read_text())
                if saved['input_sha256']!=r['files_sha256'][path.name] or sha256(target)!=saved['output_sha256']:
                    raise RuntimeError('Paired bootstrap saved values changed')
                values=np.load(target,allow_pickle=False)
            else:
                p=np.load(path,mmap_mode='r',allow_pickle=False)
                prepared=PreparedAUC.create(y.ravel(),p.ravel(),t.ravel(),codes,n)
                values=prepared.evaluate(counts)
                np.testing.assert_allclose(values[0],ts_auc(y.ravel(),p.ravel(),t.ravel()),rtol=0,atol=1e-14)
                np.save(target,values,allow_pickle=False)
                write_json(receipt,{'input_sha256':r['files_sha256'][path.name],'output_sha256':sha256(target),
                    'paired_counts_sha256':__import__('hashlib').sha256(counts.tobytes()).hexdigest()})
            matrices[eid]=values
        for eid,values in matrices.items():
            delta=values-matrices[J]
            rows.append({'candidate':eid,'reference':J,'scenario':scenario,'seed_pairs':n,
                'ts_auc':float(values[0]),'ts_auc_ci_low':float(np.quantile(values[1:],.025)),
                'ts_auc_ci_high':float(np.quantile(values[1:],.975)),
                'probability_auc_above_chance':float(np.mean(values[1:]>.5)),
                'delta':float(delta[0]),'delta_ci_low':float(np.quantile(delta[1:],.025)),
                'delta_ci_high':float(np.quantile(delta[1:],.975)),
                'probability_delta_positive':float(np.mean(delta[1:]>0)),'replicates':2000,
                'paired_control_changed_resampling':True,'artifact_kind':'frozen all-DEV model'})
        pd.DataFrame(rows).to_csv(out/'PAIRED_RESULTS.csv',index=False)
        print('NEXT2 COUNTEREXAMPLE BOOTSTRAP',scenario,round(time.perf_counter()-started,1),flush=True)
        checkpoint('Paired synthetic uncertainty '+scenario,['Remaining fixed break mechanisms','Synthetic and real-DEV tradeoff synthesis'])
    write_json(out/'RESULTS.json',{'status':'PASS','identity':identity,'created_utc':utc_now(),'input_files_sha256':sources,
        'rows':len(rows),'elapsed_seconds':time.perf_counter()-started,'seal_rows':0,
        'limitations':'Specified DGP only; conditional on fitted models; no correction for adaptive comparisons; calibration-threshold uncertainty is not included in these rank-score intervals.'})
    print('NEXT2 COUNTEREXAMPLE BOOTSTRAP COMPLETE',flush=True)


if __name__=='__main__':
    main()
