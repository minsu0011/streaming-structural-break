"""Whole-pair uncertainty for every cell of the frozen cross-background supplement."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
import pandas as pd
from numba import set_num_threads
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.bootstrap import PreparedAUC
from src.validation.group_bootstrap import draw_group_multiplicities
from src.scoring.ts_auc import ts_auc
from scripts.bootstrap_next2_counterexamples import MODELS,paired_group_codes
from src.next2.cross_background_scale import FACTORS,BACKGROUNDS
from src.next2.session import checkpoint
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    set_num_threads(1)
    source_root=ROOT/'artifacts/next2/cross_background_scale'
    source_result=json.loads((source_root/'RESULTS.json').read_text());assert source_result['status']=='PASS'
    out=ROOT/'artifacts/next2/cross_background_bootstrap';out.mkdir(parents=True,exist_ok=True)
    identity={'script_sha256':sha256(Path(__file__)),'source_result_sha256':sha256(source_root/'RESULTS.json'),
        'bootstrap_source_sha256':sha256(ROOT/'src/next/bootstrap.py'),'replicates':2000,'seed_base':2026090948,
        'seed_rule':'base plus background ordinal; same pair multiplicities across factors and models within each background',
        'resampling_unit':'control/intervention common seed pair, both complete trajectories retained','new_fits':0}
    if (out/'POLICY.json').exists():assert json.loads((out/'POLICY.json').read_text())==identity
    else:write_json(out/'POLICY.json',identity)
    rows=[];receipts={}
    for ordinal,bg in enumerate(BACKGROUNDS):
        for factor in FACTORS:
            scenario=bg+'_VAR_'+str(factor).replace('.','p');source=source_root/'evaluation'/scenario;folder=out/scenario;folder.mkdir(exist_ok=True)
            r=json.loads((source/'RESULTS.json').read_text());assert r['status']=='PASS'
            tau=np.load(source/'synthetic_tau.npy',allow_pickle=False);assert sha256(source/'synthetic_tau.npy')==r['files_sha256']['synthetic_tau.npy']
            n=len(tau)//2;h=1024;assert np.all(tau[:n]<0) and np.all(tau[n:]>=0)
            t=np.broadcast_to(np.arange(h,dtype=np.int32),(2*n,h));y=((t>=tau[:,None])&(tau[:,None]>=0)).astype(np.uint8)
            codes=paired_group_codes(n,h);_,counts=draw_group_multiplicities(np.arange(n),2000,2026090948+ordinal)
            values={}
            for eid in MODELS:
                path=source/(eid+'.npy');assert sha256(path)==r['files_sha256'][path.name]
                target=folder/(eid+'.npy');receipt=folder/(eid+'.json')
                ident={'source_sha256':sha256(path),'policy_sha256':sha256(out/'POLICY.json'),'pair_counts_sha256':hashlib.sha256(counts.tobytes()).hexdigest()}
                if receipt.exists():
                    saved=json.loads(receipt.read_text());assert saved['identity']==ident and sha256(target)==saved['output_sha256']
                    vector=np.load(target,allow_pickle=False)
                else:
                    p=np.load(path,mmap_mode='r',allow_pickle=False)
                    vector=PreparedAUC.create(y.ravel(),p.ravel(),t.ravel(),codes,n).evaluate(counts)
                    np.testing.assert_allclose(vector[0],ts_auc(y.ravel(),p.ravel(),t.ravel()),rtol=0,atol=1e-14)
                    np.save(target,vector,allow_pickle=False)
                    write_json(receipt,{'status':'PASS','identity':ident,'output_sha256':sha256(target)})
                values[eid]=vector;receipts[receipt.relative_to(ROOT).as_posix()]=sha256(receipt)
            for eid,vector in values.items():
                delta=vector-values[MODELS[0]]
                rows.append({'candidate':eid,'reference':MODELS[0],'background':bg,'variance_factor':factor,'scenario':scenario,
                    'ts_auc':float(vector[0]),'ts_auc_ci_low':float(np.quantile(vector[1:],.025)),
                    'ts_auc_ci_high':float(np.quantile(vector[1:],.975)),'probability_auc_above_chance':float(np.mean(vector[1:]>.5)),
                    'delta':float(delta[0]),'delta_ci_low':float(np.quantile(delta[1:],.025)),
                    'delta_ci_high':float(np.quantile(delta[1:],.975)),'probability_delta_positive':float(np.mean(delta[1:]>0)),
                    'replicates':2000,'seed_pairs':n,'artifact_kind':'frozen all-DEV model; prospective cross-background supplement'})
            pd.DataFrame(rows).to_csv(out/'PAIRED_RESULTS.csv',index=False)
            checkpoint('Cross-background paired uncertainty '+scenario,['Remaining fixed cells','Interpret background-conditional weakness'])
            print('NEXT2 CROSS BACKGROUND BOOTSTRAP',scenario,flush=True)
    write_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'identity':identity,'receipts_sha256':receipts,
        'table_sha256':sha256(out/'PAIRED_RESULTS.csv'),'comparisons':len(rows),'new_fits':0,
        'limitations':'Conditional on fitted models and specified DGP, not training/selection corrected. Factors share pair draws within background; cells are not independent repeats.'})


if __name__=='__main__':
    main()
