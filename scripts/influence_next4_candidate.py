"""Explain B1/B2 gains and fragile margins using fixed OOF, not another fit."""
from pathlib import Path
import sys
import argparse
import numpy as np
import pandas as pd
from numba import set_num_threads
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next4.influence import time_credits,delete_one_scores
from src.next4.frozen import fold_predictions
from src.next.data import prepare_dev


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',required=True);args=parser.parse_args()
    policy_path=BASE/'diagnostic_policy/INFLUENCE_POLICY.json';policy=read_json(policy_path)
    assert args.candidate in policy['candidates']
    for path,sha in policy['sources_sha256'].items():assert digest(ROOT/path)==sha
    source=BASE/'experiments'/args.candidate/'full';full=read_json(source/'RESULTS.json')
    out=BASE/'influence'/args.candidate;out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'START.json',{'created_utc':utc_now(),'policy_sha256':digest(policy_path),'full_result_sha256':digest(source/'RESULTS.json'),'new_fits':0},immutable=True)
    set_num_threads(2);data=prepare_dev(ROOT);metadata=data.rows()
    mp=ROOT/'artifacts/next/current_forensic/DEV_SIGNATURES.parquet'
    assert digest(mp)==read_json(mp.parent/'REGIME_ASSIGNMENT_LOCK.json')['files_sha256'][mp.name]
    series=pd.read_parquet(mp).set_index('dataset_id');data.guard.admit(series.index.to_numpy(),purpose='NEXT4 fixed-OOF influence diagnostics')
    credits=[];deletions=[];observed={};references={};new={}
    for fold in range(5):
        rows=np.flatnonzero(metadata['fold']==fold);fr=read_json(source/f'fold_{fold}.json')
        for name in (f'fold_{fold}.npy',f'fold_{fold}_global_rows.npy'):
            assert digest(source/name)==fr['files_sha256'][name]
        np.testing.assert_array_equal(np.load(source/f'fold_{fold}_global_rows.npy',allow_pickle=False),rows)
        p=np.load(source/f'fold_{fold}.npy',allow_pickle=False);frozen,proof=fold_predictions(data,fold,'full',rows)
        ids,y,t=(metadata[k][rows] for k in ('dataset_id','target','time_online'));ref=frozen['BL80'];meta=series.reindex(ids)
        credits.extend({'candidate':args.candidate,'fold':fold,**r,'contribution_to_mean_five_fold_delta':r['contribution_to_fold_delta']/5} for r in time_credits(y,p,ref,t))
        masks={'all_time':np.ones(len(y),dtype=bool),'SCALE':(meta.signature.to_numpy()=='SCALE_DOMINANT')|(meta.tau.to_numpy()<0)}
        for scope,mask in masks.items():
            progress('Exact fitted-OOF deletion sensitivity',args.candidate+' '+scope+' fold '+str(fold))
            groups,a=delete_one_scores(y[mask],ref[mask],t[mask],ids[mask]);other,b=delete_one_scores(y[mask],p[mask],t[mask],ids[mask])
            np.testing.assert_array_equal(groups,other)
            observed.setdefault(scope,[]).append(float(b[0]-a[0]));references.setdefault(scope,[]).append(float(a[0]));new.setdefault(scope,[]).append(float(b[0]))
            deleted_type=series.reindex(groups).signature.to_numpy();deleted_null=series.reindex(groups).tau.to_numpy()<0
            np.save(out/f'{scope}_fold_{fold}_paired_deletion_scores.npy',np.column_stack([a,b]),allow_pickle=False)
            for i,group in enumerate(groups):
                deletions.append({'scope':scope,'fold':fold,'diagnostic_group_ordinal':i,'removed_type':'NO_BREAK' if deleted_null[i] else str(deleted_type[i]),
                    'change_in_fold_delta':float((b[i+1]-a[i+1])-(b[0]-a[0]))})
            print('NEXT4 DELETION SENSITIVITY',args.candidate,scope,fold,len(groups),flush=True)
    summaries=[]
    for scope,deltas in observed.items():
        original=float(np.mean(deltas));records=[r for r in deletions if r['scope']==scope]
        for r in records:r['mean_five_fold_delta_after_deletion']=original+r['change_in_fold_delta']/5
        values=np.asarray([r['mean_five_fold_delta_after_deletion'] for r in records])
        summaries.append({'scope':scope,'observed_mean_delta':original,'deleted_groups':len(values),
            'minimum_mean_delta_after_one_deletion':float(values.min()),'maximum_mean_delta_after_one_deletion':float(values.max()),
            'fraction_deletions_delta_positive':float(np.mean(values>0)),
            'fraction_deletions_above_original_numeric_margin':float(np.mean(values>=(.0013 if scope=='all_time' else .002))),
            'new_CI_or_gate_substitution':False})
    frame=pd.DataFrame(credits);frame.to_csv(out/'TIME_PAIR_CREDITS.csv',index=False)
    frame.groupby('time_bin',sort=False,as_index=False).agg(contribution_to_mean_delta=('contribution_to_mean_five_fold_delta','sum'),
        mean_pair_mass_fraction=('fraction_of_fold_pair_mass','mean')).to_csv(out/'TIME_PAIR_CREDIT_SUMMARY.csv',index=False)
    pd.DataFrame(deletions).to_csv(out/'DELETION_DIAGNOSTICS.csv',index=False)
    np.testing.assert_allclose(frame.contribution_to_mean_five_fold_delta.sum(),full['mean_delta'],rtol=0,atol=1e-14)
    data.guard.check_state()
    files={p.name:digest(p) for p in out.iterdir() if p.is_file()}
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'candidate':args.candidate,'new_fits':0,'seal_rows':0,
        'deletion_summaries':summaries,'exact_time_credit_sum':float(frame.contribution_to_mean_five_fold_delta.sum()),
        'interpretation':'Each deletion removes one held-out whole series from its existing fold while keeping predictions fitted and all other folds fixed. This is sensitivity, not leave-one-out model refitting or a replacement CI. Group ordinals are diagnostic-only and do not enter inference.',
        'files_sha256':files},immutable=True)
    print('NEXT4 INFLUENCE DIAGNOSTICS COMPLETE',args.candidate,flush=True)


if __name__=='__main__':main()
