"""Supplement inherited slice membership counts without changing scores or gates."""
from pathlib import Path
import sys
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next4.diagnostics import evaluation_masks,scale_directions
from src.next4.support import scored_support
from src.next.data import prepare_dev


def main():
    out=BASE/'scored_support';out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'START.json',{'created_utc':utc_now(),'source_sha256':digest(Path(__file__)),
        'question':'How many series actually contribute positive observations and eligible scoring pairs in each inherited diagnostic slice?',
        'new_fits':0,'new_model_scores':0,'score_or_gate_changes':False},immutable=True)
    data=prepare_dev(ROOT);metadata=data.rows();path=ROOT/'artifacts/next/current_forensic/DEV_SIGNATURES.parquet'
    assert digest(path)==read_json(path.parent/'REGIME_ASSIGNMENT_LOCK.json')['files_sha256'][path.name]
    series=pd.read_parquet(path).set_index('dataset_id');data.guard.admit(series.index.to_numpy(),purpose='NEXT4 scored support audit')
    directions,proof=scale_directions(data,series);records=[]
    inherited=pd.read_csv(BASE/'forensic/B2_LOG_ENERGY_SEMITAIL/SUPPORT.csv')
    for fold in range(5):
        rows=np.flatnonzero(metadata['fold']==fold);ids,y,t=(metadata[k][rows] for k in ('dataset_id','target','time_online'))
        meta=series.reindex(ids);tau=meta.tau.to_numpy()
        for scope,mask in evaluation_masks(meta,y,t,directions).items():
            r=scored_support(ids[mask],y[mask],t[mask],tau[mask]);original=inherited[(inherited.fold==fold)&(inherited.scope==scope)].iloc[0]
            assert r['break_series']==original.break_series and r['null_series']==original.null_series and r['rows']==original.points
            records.append({'scope':scope,'fold':fold,**r})
        progress('Actual scoring support by slice','Original fold '+str(fold)+' membership, observed positive series and eligible pair mass reconciled')
    frame=pd.DataFrame(records);frame.to_csv(out/'FOLD_SCORED_SUPPORT.csv',index=False)
    totals=frame.drop(columns='fold').groupby('scope',sort=False,as_index=False).sum()
    totals['at_least_100_scoring_positive_series']=totals.eligible_positive_series>=100
    totals.to_csv(out/'SCORED_SUPPORT.csv',index=False);data.guard.check_state()
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'new_fits':0,'new_model_scores':0,'seal_rows':0,
        'inherited_membership_counts_exact':True,'original_scores_bootstraps_and_gates_unchanged':True,
        'interpretation':'Inherited break_series counts eventual-break trajectories present in the slice. Time and age slices can include their pre-change negative observations. Report positive_series and eligible_positive_series separately; an eventual-break membership count alone is not the number of informative positives.',
        'rows':totals.to_dict('records'),'directional_scope':proof,
        'files_sha256':{name:digest(out/name) for name in ('FOLD_SCORED_SUPPORT.csv','SCORED_SUPPORT.csv')}},immutable=True)
    print('NEXT4 SCORED SUPPORT AUDIT COMPLETE',flush=True)


if __name__=='__main__':main()
