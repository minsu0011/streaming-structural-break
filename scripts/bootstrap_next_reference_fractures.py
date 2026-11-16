"""Whole-series paired uncertainty on fixed time/age/regime/mechanism slices."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from numba import set_num_threads
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.guard import SealGuard
from src.next.references import REFERENCES, load_reference_fold, verify_reference_sources
from src.next.bootstrap import PreparedAUC, paired_summary
from src.validation.group_bootstrap import draw_group_multiplicities
from src.next.progress import checkpoint
from src.utils.artifacts import write_json, sha256, utc_now


def main():
    set_num_threads(3)
    started = time.perf_counter()
    guard = SealGuard(ROOT)
    verify_reference_sources(ROOT)
    root_policy = json.loads((ROOT/'ROBUSTNESS_SPLIT_LOCK.json').read_text(encoding='utf-8'))
    bootstrap = root_policy['bootstrap']
    metadata_path = ROOT/'artifacts/next/current_forensic/DEV_SIGNATURES.parquet'
    metadata = pd.read_parquet(metadata_path).set_index('dataset_id')
    guard.admit(metadata.index.to_numpy(), purpose='paired fracture diagnostics')
    out = ROOT/'artifacts/next/reference_fracture_bootstrap'
    out.mkdir(exist_ok=True)
    policy = {'status':'LOCKED','replicates':2000,'seed':bootstrap['seed'],
        'sampling':'same full DEV within-fold whole-series multiplicities before applying every fixed slice',
        'scope':'post-hoc conditional fixed-OOF diagnostic; not model-selection-adjusted inference',
        'regime_assignment_sha256':sha256(metadata_path),'split_sha256':guard.split.digest,
        'bootstrap_source_sha256':sha256(ROOT/'src/next/bootstrap.py'),'script_sha256':sha256(Path(__file__))}
    if (out/'POLICY.json').exists() and json.loads((out/'POLICY.json').read_text(encoding='utf-8')) != policy:
        raise RuntimeError('Fracture bootstrap policy changed')
    write_json(out/'POLICY.json',policy)
    if (out/'RESULTS.json').exists():
        print('Reference fracture bootstrap already complete',flush=True)
        return
    all_scores = {role:{} for role in REFERENCES}
    supports = []
    for fold in range(5):
        frame, primary, _ = load_reference_fold(ROOT,guard,'PRIMARY',fold)
        ids,y,t = frame.dataset_id.to_numpy(),frame.target.to_numpy(),frame.time_online.to_numpy()
        rowmeta = metadata.reindex(ids)
        age = t-rowmeta.tau.to_numpy()
        code,counts = draw_group_multiplicities(ids,2000,bootstrap['seed']+fold)
        masks = {}
        for name,a,b in [('0-9',0,10),('10-24',10,25),('25-49',25,50),('50-99',50,100),('100-199',100,200),('200-399',200,400),('400+',400,100000)]:
            masks['time_'+name] = (t>=a)&(t<b)
        for name,a,b in [('0-4',0,5),('5-9',5,10),('10-19',10,20),('20-49',20,50),('50+',50,100000)]:
            masks['age_'+name] = (y==0)|((y==1)&(age>=a)&(age<b))
        for regime in range(5):
            masks['regime_'+str(regime)] = rowmeta.historical_regime.to_numpy()==regime
        for signature in root_policy['mechanism'].get('names', ['MEAN_DOMINANT','SCALE_DOMINANT','DEPENDENCE_DOMINANT','TAIL_DOMINANT','MIXED','WEAK_OR_AMBIGUOUS']):
            masks['mechanism_'+signature] = np.isin(rowmeta.signature.to_numpy(),[signature,'NO_BREAK_CONTROL'])
        for label,mask in masks.items():
            supports.append({'fold':fold,'scope':label,'series':len(np.unique(ids[mask])),
                             'positive_rows':int(y[mask].sum()),'rows':int(mask.sum())})
        for role in REFERENCES:
            if role == 'PRIMARY':
                prediction = primary
            else:
                other,prediction,_ = load_reference_fold(ROOT,guard,role,fold)
                pd.testing.assert_frame_equal(frame,other)
            for label,mask in masks.items():
                path = out/f'{role}_{fold}_{label}.npy'
                if path.exists():
                    samples = np.load(path,allow_pickle=False)
                else:
                    samples = PreparedAUC.create(y[mask],prediction[mask],t[mask],code[mask],counts.shape[1]).evaluate(counts)
                    np.save(path,samples,allow_pickle=False)
                all_scores[role].setdefault(label,[]).append(samples)
            print('FRACTURE BOOTSTRAP',role,fold,round(time.perf_counter()-started,1),'seconds',flush=True)
        write_json(out/'PROGRESS.json',{'completed_folds':fold+1,'timestamp':utc_now()})
    rows = []
    for label in all_scores['PRIMARY']:
        primary = np.asarray(all_scores['PRIMARY'][label]).T
        for role in ['HIGHEST_MEAN','ROBUST_BACKUP']:
            candidate = np.asarray(all_scores[role][label]).T
            rows.append({'scope':label,'reference':'PRIMARY','candidate':role,
                'reference_mean':float(primary[0].mean()),'candidate_mean':float(candidate[0].mean()),
                **paired_summary(primary,candidate)})
    pd.DataFrame(rows).to_csv(out/'PAIRED_FRACTURES.csv',index=False)
    pd.DataFrame(supports).to_csv(out/'SUPPORT.csv',index=False)
    write_json(out/'RESULTS.json',{'status':'PASS','timestamp':utc_now(),'policy':policy,
        'elapsed_seconds':time.perf_counter()-started,'comparisons':len(rows),'seal_rows':0,'reduced_new_usage':0,
        'files_sha256':{p.name:sha256(p) for p in out.glob('*.npy')},
        'limitations':'Intervals are not adjusted for multiple slices, adaptive candidate selection or fitting uncertainty. Sparse slices can have no positive-negative pairs in a resample; the official 0.5 fallback is retained.'})
    checkpoint(ROOT,'reference-fracture-uncertainty',['46 paired 2000-replicate fracture comparisons'],
        ['finite initial candidate queue','historical-null feature calibration'],
        artifacts=['artifacts/next/reference_fracture_bootstrap/PAIRED_FRACTURES.csv'],best=.611292686371357)
    print('REFERENCE FRACTURE BOOTSTRAP COMPLETE',round(time.perf_counter()-started,1),'seconds',flush=True)


if __name__ == '__main__':
    main()
