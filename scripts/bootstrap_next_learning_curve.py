"""Paired fixed-fit uncertainty for the completed matched learning curves."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from numba import set_num_threads
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.data import prepare_dev
from src.next.bootstrap import PreparedAUC,paired_summary
from src.next.bootstrap_integrity import load_reference_samples
from src.validation.group_bootstrap import draw_group_multiplicities
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    set_num_threads(2)
    data = prepare_dev(ROOT)
    metadata = data.rows()
    folder = ROOT/'artifacts/next/learning_curve'
    original = json.loads((folder/'RESULTS.json').read_text(encoding='utf-8'))
    if original['status']!='PASS' or original['new_partial_training_fits']!=45:
        raise RuntimeError('Complete fixed learning curves required')
    policy = json.loads((ROOT/'ROBUSTNESS_SPLIT_LOCK.json').read_text(encoding='utf-8'))['bootstrap']
    out = folder/'paired_bootstrap'
    out.mkdir(exist_ok=True)
    identity = {'learning_result_sha256':sha256(folder/'RESULTS.json'),
        'learning_policy_sha256':sha256(ROOT/'configs/next_learning_curve_policy.json'),
        'script_sha256':sha256(Path(__file__)),'replicates':policy['final_replicates'],'seed':policy['seed'],
        'interpretation':'Same original held-out series and multiplicities at every nested training fraction; fixed-fit conditional uncertainty, not adaptive-selection or training-fit uncertainty.'}
    manifest = out/'POLICY.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8'))!=identity:
        raise RuntimeError('Learning-curve bootstrap identity changed')
    if not manifest.exists():
        write_json(manifest,identity)
    results = {}
    for fold in range(5):
        rows = np.flatnonzero(metadata['fold']==fold)
        ids,y,t = (metadata[k][rows] for k in ('dataset_id','target','time_online'))
        codes,counts = draw_group_multiplicities(ids,policy['final_replicates'],policy['seed']+fold)
        for role in ('PRIMARY','HIGHEST_MEAN','V007_VARIANCE_FORECAST'):
            for fraction in (25,50,75):
                stem = f'fold_{fold}_fraction_{fraction}'
                record = json.loads((folder/role/(stem+'.json')).read_text(encoding='utf-8'))
                for name,digest in record['files_sha256'].items():
                    if sha256(folder/role/name)!=digest:
                        raise RuntimeError('Learning-curve source model or predictions changed')
                path = out/f'{role}_{fold}_{fraction}.npy'
                digest_path = path.with_suffix('.json')
                if path.exists() and digest_path.exists():
                    if sha256(path)!=json.loads(digest_path.read_text(encoding='utf-8'))['sha256']:
                        raise RuntimeError('Partial learning-curve bootstrap changed')
                    values = np.load(path,allow_pickle=False)
                else:
                    prediction = np.load(folder/role/(stem+'.npy'),allow_pickle=False)
                    values = PreparedAUC.create(y,prediction,t,codes,counts.shape[1]).evaluate(counts)
                    np.save(path,values,allow_pickle=False)
                    write_json(digest_path,{'sha256':sha256(path),'created_utc':utc_now()})
                np.testing.assert_allclose(values[0],record['ts_auc'],rtol=0,atol=1e-14)
                results.setdefault((role,fraction),[]).append(values)
            if role=='V007_VARIANCE_FORECAST':
                p = ROOT/'artifacts/next/forensic'/role/f'fold_{fold}_all_time.npy'
                receipt = json.loads((p.parent/'RESULTS.json').read_text(encoding='utf-8'))
                if sha256(p)!=receipt['files_sha256'][p.name]:
                    raise RuntimeError('Full-size variance bootstrap changed')
                full = np.load(p,allow_pickle=False)
            else:
                full = load_reference_samples(ROOT,role,fold)
            results.setdefault((role,100),[]).append(full)
        print('LEARNING BOOTSTRAP',fold,flush=True)
    rows = []
    for fraction in (25,50,75,100):
        new = np.asarray(results[('V007_VARIANCE_FORECAST',fraction)]).T
        for role in ('PRIMARY','HIGHEST_MEAN'):
            old = np.asarray(results[(role,fraction)]).T
            rows.append({'candidate':'V007_VARIANCE_FORECAST','reference':role,'training_fraction':fraction/100,
                'candidate_mean':float(new[0].mean()),'reference_mean':float(old[0].mean()),**paired_summary(old,new)})
    pd.DataFrame(rows).to_csv(out/'PAIRED_RESULTS.csv',index=False)
    write_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'comparisons':rows,'identity':identity,
        'seal_rows':0,'reduced_new_usage':0,'selection_authority':False})
    print('LEARNING BOOTSTRAP COMPLETE',flush=True)


if __name__=='__main__':
    main()
