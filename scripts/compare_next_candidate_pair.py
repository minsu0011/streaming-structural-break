"""Check two full-CV candidates for complementary supported fractures.

This script never forms an averaged prediction. Late combinations require a
separate prospectively frozen weight grid and the elapsed-time admission gate.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.data import prepare_dev
from src.next.candidate_io import load_full_oof
from src.next.bootstrap import paired_summary
from scripts.forensic_next_candidate import rank_within_time
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--left',default='V007_VARIANCE_FORECAST')
    parser.add_argument('--right',default='Q005_VARIANCE_NO_MEMORY')
    args = parser.parse_args()
    data = prepare_dev(ROOT)
    left,lp,metadata,lvalue,lsources = load_full_oof(data,args.left)
    right,rp,rmetadata,rvalue,rsources = load_full_oof(data,args.right)
    for name in metadata:
        np.testing.assert_array_equal(metadata[name],rmetadata[name])
    paths = [ROOT/'artifacts/next/forensic'/eid for eid in (args.left,args.right)]
    reports = [json.loads((path/'RESULTS.json').read_text(encoding='utf-8')) for path in paths]
    policies = [json.loads((path/'POLICY.json').read_text(encoding='utf-8')) for path in paths]
    for key in ('replicates','policy_sha256','series_metadata_sha256','reference_fracture_result_sha256','script_sha256'):
        if policies[0][key]!=policies[1][key]:
            raise RuntimeError('Pair forensic policies do not match')
    for path,report in zip(paths,reports):
        if report['status']!='PASS' or report['all_time_paired_bootstrap']['replicates']!=2000:
            raise RuntimeError('Complete 2000-replicate forensic reports required')
        for name,digest in report['files_sha256'].items():
            if sha256(path/name)!=digest:
                raise RuntimeError('Pair forensic artifact changed')
    rows = []
    for item in pd.read_csv(paths[0]/'PAIRED_FRACTURES.csv').to_dict('records'):
        scope = item['scope']
        ldraws = np.asarray([np.load(paths[0]/f'fold_{fold}_{scope}.npy',allow_pickle=False) for fold in range(5)]).T
        rdraws = np.asarray([np.load(paths[1]/f'fold_{fold}_{scope}.npy',allow_pickle=False) for fold in range(5)]).T
        rows.append({'left':args.left,'right':args.right,'scope':scope,'supported':item['supported'],
            'break_series':item['break_series'],'null_series':item['null_series'],
            'left_mean':float(ldraws[0].mean()),'right_mean':float(rdraws[0].mean()),**paired_summary(rdraws,ldraws)})
    correlations = []
    disagreements = []
    for fold in range(5):
        mask = metadata['fold']==fold
        lrank = rank_within_time(lvalue[mask],metadata['time_online'][mask])
        rrank = rank_within_time(rvalue[mask],metadata['time_online'][mask])
        correlations.append({'fold':fold,'within_time_rank_correlation':float(np.corrcoef(lrank,rrank)[0,1]),
            'raw_prediction_correlation':float(np.corrcoef(lvalue[mask],rvalue[mask])[0,1])})
        frame = pd.DataFrame({'dataset_id':metadata['dataset_id'][mask],'rank_difference':lrank-rrank,
            'absolute_rank_difference':np.abs(lrank-rrank),'absolute_score_difference':np.abs(lvalue[mask]-rvalue[mask])})
        summary = frame.groupby('dataset_id').agg(mean_rank_difference=('rank_difference','mean'),
            mean_absolute_rank_difference=('absolute_rank_difference','mean'),mean_absolute_score_difference=('absolute_score_difference','mean'))
        summary['fold']=fold
        disagreements.append(summary.reset_index())
    frame = pd.DataFrame(rows)
    supported = frame.loc[frame.supported & frame.scope.str.startswith(('mechanism_','regime_'))]
    correlation = float(np.mean([r['within_time_rank_correlation'] for r in correlations]))
    complementarity = {'mean_within_time_rank_correlation':correlation,'correlation_threshold':.98,
        'supported_left_gain_at_least_0_002':bool((supported.estimate_mean_delta>=.002).any()),
        'supported_right_gain_at_least_0_002':bool((supported.estimate_mean_delta<=-.002).any())}
    complementarity['eligible_for_predeclared_late_combination'] = bool(correlation<.98 and complementarity['supported_left_gain_at_least_0_002'] and complementarity['supported_right_gain_at_least_0_002'])
    out = ROOT/'artifacts/next/pair_complementarity'/(args.left+'__'+args.right)
    out.mkdir(parents=True,exist_ok=True)
    frame.to_csv(out/'PAIRED_FRACTURES.csv',index=False)
    pd.DataFrame(correlations).to_csv(out/'CORRELATIONS.csv',index=False)
    pd.concat(disagreements,ignore_index=True).to_csv(out/'DISAGREEMENTS.csv',index=False)
    write_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'left':args.left,'right':args.right,
        'left_oof_sources':lsources,'right_oof_sources':rsources,'complementarity':complementarity,
        'script_sha256':sha256(Path(__file__)),'paired_all_time':rows[0],
        'averaged_predictions_formed':False,'averaged_prediction_scores_inspected':False,
        'seal_rows':0,'reduced_new_usage':0,
        'interpretation':'Pair nomination follows independently observed mechanism/regime tradeoffs. Correlation and opposite supported gains use the existing .98 and +/- .002 eligibility thresholds; any weight grid must be separately frozen before late scoring.'})
    print('PAIR COMPLEMENTARITY',args.left,args.right,complementarity,flush=True)


if __name__=='__main__':
    main()
