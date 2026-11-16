"""Frozen, conservative complementarity audit before any late combination."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.data import prepare_dev
from src.next.guard import SealGuard
from src.scoring.ts_auc import ts_auc
from scripts.forensic_next import ranked_within_time
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    data=prepare_dev(ROOT)
    root_policy=json.loads((ROOT/'ROBUSTNESS_SPLIT_LOCK.json').read_text(encoding='utf-8'))
    out=ROOT/'artifacts/next/screen_complementarity'
    out.mkdir(exist_ok=True)
    policy={'status':'LOCKED','threshold_rank_correlation_max':.98,'minimum_opposite_direction_effect':.002,
        'mechanism_minimum_break_series_in_evaluated_screen':100,
        'regime_minimum_break_and_no_break_series_each':100,
        'unit':'aggregated fixed screen validation folds 0 and 4; scoring averages the two original folds',
        'excluded':'no unadmitted candidate full-CV predictions, seal values or reduced values',
        'partners':['N001_SEQ_AR8_SD_CORE','N005_WHITEN_AR12','N020_STANDALONE_S2'],
        'purpose':'diagnose complementary errors before a separately preregistered fixed combination',
        'screen_policy_sha256':sha256(ROOT/'ROBUSTNESS_SPLIT_LOCK.json')}
    policy_path=out/'POLICY.json'
    if policy_path.exists() and json.loads(policy_path.read_text(encoding='utf-8'))!=policy:
        raise RuntimeError('Complementarity policy changed')
    write_json(policy_path,policy)
    if (out/'RESULTS.json').exists():
        print('Complementarity audit already complete')
        return
    metadata=pd.read_parquet(ROOT/'artifacts/next/current_forensic/DEV_SIGNATURES.parquet').set_index('dataset_id')
    rows,correlations=[],[]
    for fold in root_policy['screen']['folds']:
        path=ROOT/'artifacts/next/screen_reference'
        coords=pd.read_parquet(path/f'fold_{fold}_coordinates.parquet')
        data.guard.record_rows(coords.dataset_id.to_numpy(),purpose='screen complementarity')
        primary=np.load(path/f'fold_{fold}.npy',allow_pickle=False)
        meta=metadata.reindex(coords.dataset_id)
        ids,y,t=coords.dataset_id.to_numpy(),coords.target.to_numpy(),coords.time_online.to_numpy()
        prank=ranked_within_time(primary,t)
        for partner in policy['partners']:
            pred=np.load(ROOT/'artifacts/next/experiments'/partner/'screen'/f'fold_{fold}.npy',allow_pickle=False)
            crank=ranked_within_time(pred,t)
            correlations.append({'partner':partner,'fold':fold,'within_time_rank_correlation':float(np.corrcoef(prank,crank)[0,1])})
            for kind in ['mechanism','regime']:
                values=['MEAN_DOMINANT','SCALE_DOMINANT','DEPENDENCE_DOMINANT','TAIL_DOMINANT','MIXED','WEAK_OR_AMBIGUOUS'] if kind=='mechanism' else list(range(5))
                for value in values:
                    if kind=='mechanism':
                        mask=np.isin(meta.signature.to_numpy(),[value,'NO_BREAK_CONTROL'])
                        member=(meta.signature.to_numpy()==value)
                    else:
                        mask=meta.historical_regime.to_numpy()==value
                        member=mask&(meta.tau.to_numpy()>=0)
                    nbreak=len(np.unique(ids[member]))
                    nnull=len(np.unique(ids[mask&(meta.tau.to_numpy()<0)]))
                    baseline=ts_auc(y[mask],primary[mask],t[mask])
                    candidate=ts_auc(y[mask],pred[mask],t[mask])
                    rows.append({'partner':partner,'fold':fold,'kind':kind,'slice':str(value),'break_series':nbreak,
                        'null_series':nnull,'primary_score':baseline,'candidate_score':candidate,'delta':candidate-baseline})
    frame=pd.DataFrame(rows)
    summary=frame.groupby(['partner','kind','slice']).agg(delta=('delta','mean'),break_series=('break_series','sum'),null_series=('null_series','sum')).reset_index()
    summary['supported']=(summary.break_series>=100)&((summary.kind=='mechanism')|(summary.null_series>=100))
    corr=pd.DataFrame(correlations).groupby('partner').within_time_rank_correlation.mean()
    decisions=[]
    for partner,group in summary.groupby('partner'):
        supported=group.loc[group.supported]
        positive=bool((supported.delta>=.002).any())
        negative=bool((supported.delta<=-.002).any())
        decisions.append({'partner':partner,'within_time_rank_correlation':float(corr[partner]),
            'supported_positive_slice':positive,'supported_negative_slice':negative,
            'complementarity_admitted':bool(corr[partner]<.98 and positive and negative),
            'score_scope':'screen only','new_full_cv_used':False})
    frame.to_csv(out/'FOLD_FRACTURES.csv',index=False)
    summary.to_csv(out/'SUMMARY.csv',index=False)
    pd.DataFrame(correlations).to_csv(out/'RANK_CORRELATIONS.csv',index=False)
    write_json(out/'RESULTS.json',{'status':'PASS','timestamp':utc_now(),'policy':policy,'decisions':decisions,
        'seal_rows':0,'reduced_new_usage':0,'limitation':'Complementarity is descriptive and does not establish ensemble benefit; every combination must independently pass the unchanged screen and full gates.'})
    print(pd.DataFrame(decisions).to_string(index=False))
    print(summary.loc[summary.supported].to_string(index=False))


if __name__=='__main__':
    main()
