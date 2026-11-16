"""Paired mechanism contrasts against the exactly matched old 51-feature fit."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.bootstrap import paired_summary
from src.next.bootstrap_integrity import load_reference_samples
from src.next.references import verify_reference_sources
from src.next.guard import SealGuard
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate',default='V007_VARIANCE_FORECAST')
    args = parser.parse_args()
    verify_reference_sources(ROOT)
    SealGuard(ROOT).check_state()
    folder = ROOT/'artifacts/next/forensic'/args.candidate
    report = json.loads((folder/'RESULTS.json').read_text(encoding='utf-8'))
    if report['status']!='PASS' or report['all_time_paired_bootstrap']['replicates']!=2000:
        raise RuntimeError('Complete final bootstrap forensic report required')
    for name,digest in report['files_sha256'].items():
        if sha256(folder/name)!=digest:
            raise RuntimeError('Candidate forensic artifact changed')
    refdir = ROOT/'artifacts/next/reference_fracture_bootstrap'
    reference = json.loads((refdir/'RESULTS.json').read_text(encoding='utf-8'))
    summaries = pd.read_csv(folder/'PAIRED_FRACTURES.csv')
    rows = []
    for item in summaries.to_dict('records'):
        scope = item['scope']
        new,old = [],[]
        for fold in range(5):
            new.append(np.load(folder/f'fold_{fold}_{scope}.npy',allow_pickle=False))
            if scope=='all_time':
                old.append(load_reference_samples(ROOT,'HIGHEST_MEAN',fold))
            else:
                name = f'HIGHEST_MEAN_{fold}_{scope}.npy'
                if sha256(refdir/name)!=reference['files_sha256'][name]:
                    raise RuntimeError('Matched baseline bootstrap changed')
                old.append(np.load(refdir/name,allow_pickle=False))
        new,old = np.asarray(new).T,np.asarray(old).T
        rows.append({'candidate':args.candidate,'reference':'HIGHEST_MEAN','scope':scope,
            'supported':item['supported'],'break_series':item['break_series'],'null_series':item['null_series'],
            'candidate_mean':float(new[0].mean()),'reference_mean':float(old[0].mean()),**paired_summary(old,new)})
    out = ROOT/'artifacts/next/matched_contrast'/args.candidate
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'PAIRED_FRACTURES.csv',index=False)
    write_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'candidate':args.candidate,
        'reference':'HIGHEST_MEAN','comparisons':len(rows),'all_time':rows[0],
        'baseline_parity_sha256':sha256(ROOT/'artifacts/next/REFERENCE_ADAPTER_PARITY.json'),
        'candidate_forensic_sha256':sha256(folder/'RESULTS.json'),'reference_forensic_sha256':sha256(refdir/'RESULTS.json'),
        'script_sha256':sha256(Path(__file__)),'seal_rows':0,'reduced_new_usage':0,
        'matched_interpretation':'Same old 51 ABCD features, binary LightGBM capacity, training order and W3 weighting; new statistics are the intervention. The exact old five-fold adapter parity is the engineering anchor.',
        'limitation':'Paired fixed-OOF inference, conditional on adaptively selected fitted models and not corrected for multiple candidate or subgroup inspection.'})
    print('MATCHED CONTRAST PASS',args.candidate,rows[0]['estimate_mean_delta'],rows[0]['probability_delta_positive'],flush=True)


if __name__=='__main__':
    main()
