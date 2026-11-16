"""Conditional whole-series uncertainty versus frozen 80/20, 24 locked scopes."""
from pathlib import Path
import sys
import argparse
import numpy as np
import pandas as pd
from numba import set_num_threads
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next3.artifacts import BASE,atomic_json,read_json,digest,utc_now,progress
from src.next3.evaluation import load_full_oof
from src.next.data import prepare_dev
from src.next.bootstrap import PreparedAUC,paired_summary
from src.validation.group_bootstrap import draw_group_multiplicities
from scripts.forensic_next2_candidate import masks_for


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',required=True);args=parser.parse_args()
    set_num_threads(2)
    data=prepare_dev(ROOT)
    spec,report,metadata,prediction,sources=load_full_oof(data,args.candidate)
    if not report.get('finalist_admitted'):raise RuntimeError('Full result not eligible for finalist diagnostics')
    out=BASE/'forensic'/args.candidate;out.mkdir(parents=True,exist_ok=True)
    if (out/'RESULTS.json').exists():raise RuntimeError('Do not overwrite completed forensic')
    path=ROOT/'artifacts/next/current_forensic/DEV_SIGNATURES.parquet'
    lock=read_json(path.parent/'REGIME_ASSIGNMENT_LOCK.json')
    assert digest(path)==lock['files_sha256'][path.name]
    series=pd.read_parquet(path).set_index('dataset_id')
    data.guard.admit(series.index.to_numpy(),purpose='NEXT3 evaluation-only diagnostic labels')
    reference=ROOT/'artifacts/next2/forensic/BL2_C205_DEPENDENCE_80'
    reference_receipt=read_json(reference/'RESULTS.json')
    policy=read_json(ROOT/'configs/next3/RESEARCH_POLICY_LOCK.json')
    identity={'candidate':args.candidate,'created_utc':utc_now(),'prediction_sources':sources,
        'reference_receipt_sha256':digest(reference/'RESULTS.json'),'metadata_sha256':digest(path),
        'policy_sha256':digest(ROOT/'configs/next3/RESEARCH_POLICY_LOCK.json'),
        'script_sha256':digest(Path(__file__)),'replicates':2000,'seed_base':policy['bootstrap']['seed_base'],
        'unit':'whole DEV series; fixed OOF and repeated adaptive selection not corrected'}
    atomic_json(out/'POLICY.json',identity,immutable=True)
    samples,refs,supports,fold_scores={},{},[],[]
    for fold in range(5):
        selected=np.flatnonzero(metadata['fold']==fold)
        ids,y,t=(metadata[k][selected] for k in ('dataset_id','target','time_online'))
        p=prediction[selected];smeta=series.reindex(ids)
        code,counts=draw_group_multiplicities(ids,2000,policy['bootstrap']['seed_base']+fold)
        for scope,mask in masks_for(smeta,y,t).items():
            filename=f'fold_{fold}_{scope}.npy'
            values=PreparedAUC.create(y[mask],p[mask],t[mask],code[mask],counts.shape[1]).evaluate(counts)
            np.save(out/filename,values,allow_pickle=False)
            assert digest(reference/filename)==reference_receipt['files_sha256'][filename]
            baseline=np.load(reference/filename,allow_pickle=False)
            samples.setdefault(scope,[]).append(values);refs.setdefault(scope,[]).append(baseline)
            supports.append({'scope':scope,'fold':fold,
                'break_series':len(np.unique(ids[mask&(smeta.tau.to_numpy()>=0)])),
                'null_series':len(np.unique(ids[mask&(smeta.tau.to_numpy()<0)]))})
            fold_scores.append({'candidate':args.candidate,'fold':fold,'scope':scope,
                'ts_auc':values[0],'reference_ts_auc':baseline[0],'delta':values[0]-baseline[0]})
        progress('NEXT3 conditional bootstrap',f'{args.candidate}: fold {fold}, 24 fixed scopes complete',experiment=args.candidate)
        print('NEXT3 FORENSIC',args.candidate,fold,flush=True)
    totals=pd.DataFrame(supports).groupby('scope')[['break_series','null_series']].sum()
    rows=[]
    for scope in samples:
        new=np.asarray(samples[scope]).T;old=np.asarray(refs[scope]).T
        rows.append({'candidate':args.candidate,'reference':'BL2_C205_DEPENDENCE_80','scope':scope,
            'candidate_mean':float(new[0].mean()),'reference_mean':float(old[0].mean()),
            'candidate_median':float(np.median(new[0])),'candidate_worst':float(new[0].min()),
            'supported':bool(totals.loc[scope,'break_series']>=100 and (not scope.startswith('regime_') or totals.loc[scope,'null_series']>=100)),
            **totals.loc[scope].astype(int).to_dict(),**paired_summary(old,new)})
    pd.DataFrame(rows).to_csv(out/'PAIRED_FRACTURES.csv',index=False)
    pd.DataFrame(fold_scores).to_csv(out/'FOLD_SCORES.csv',index=False)
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'candidate':args.candidate,
        'scopes':len(rows),'replicates':2000,'new_fits':0,'seal_rows':0,
        'all_time':next(r for r in rows if r['scope']=='all_time'),
        'files_sha256':{p.name:digest(p) for p in out.iterdir() if p.is_file()},
        'conditional_not_selection_adjusted':True},immutable=True)
    print('NEXT3 FORENSIC COMPLETE',args.candidate,flush=True)


if __name__=='__main__':main()
