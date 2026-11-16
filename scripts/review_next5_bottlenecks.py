"""Evidence-based board snapshots; never generate or mutate scientific recipes."""
from pathlib import Path
import argparse
import csv
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next5.artifacts import BASE,atomic_json,read_json,digest,utc_now,progress
from src.next5.saturation import assert_budget


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reason',required=True);parser.add_argument('--active',default=None);args=parser.parse_args()
    previous=read_json(BASE/'bottlenecks/BOARD.json');rows=[dict(row) for row in previous['rows']]
    full=[read_json(p) for p in sorted((BASE/'experiments').glob('*/full/RESULTS.json'))]
    screens=[read_json(p) for p in sorted((BASE/'experiments').glob('*/screen/RESULTS.json'))]
    bootstrap=[read_json(p) for p in sorted((BASE/'bootstrap').glob('*/RESULTS.json'))]
    now=utc_now();saturation=read_json(BASE/'runtime/SATURATION.json')
    active=set(args.active.split(',')) if args.active else None
    if active is not None and (len(active)>2 or not active.issubset({row['bottleneck'] for row in rows})):raise ValueError('At most two known active bottlenecks required')
    finite_menu_complete=len(full)+sum(r.get('screen_pass') is False for r in screens)==20
    for row in rows:
        row['timestamp']=now;families=set(row['families'].split(','))
        completed=full if 'ALL' in families else [r for r in full if r['family'] in families]
        row['NEXT5_completed_full']=len(completed)
        row['NEXT5_measured_gain']=max((r['mean_delta'] for r in completed),default=None)
        row['NEXT5_measured_results']='; '.join(r['candidate']+' '+format(r['mean_delta'],'.7f') for r in completed)
        row['NEXT5_primary_uncertainty_admitted']='; '.join(r['candidate'] for r in bootstrap if r.get('performance_primary_and_bootstrap_admission') and (r['family'] in families or 'ALL' in families))
        if row['bottleneck']=='B1' and completed and max(r['mean_delta'] for r in completed)<.0012:
            row['estimated_TS_AUC_impact_subjective_prior']=.0012
            row['latest_evidence_note']='Completed direct scale candidates do not establish the common promotion threshold.'
        if saturation.get('stop_new_scientific_fits'):
            row['status']='ACTIVE' if row['bottleneck'] in ('B4','B6') else 'CLOSED_NEW_FITS'
        if active is not None:
            row['status']='ACTIVE' if row['bottleneck'] in active else ('CLOSED_NEW_HYPOTHESES' if finite_menu_complete else 'MONITOR')
        row['priority_score']=row['estimated_TS_AUC_impact_subjective_prior']*row['evidence_prior_1_to_5']*row['generalization_relevance_1_to_5']/(row['overfit_risk_1_to_5']*row['implementation_complexity_1_to_5']*row['runtime_cost_1_to_5'])
    assert sum(row['status']=='ACTIVE' for row in rows)<=2
    failures=[read_json(p) for p in (BASE/'jobs').glob('*/STATE.json') if read_json(p)['status'] in ('FAILED','QUARANTINED')]
    accounting=read_json(BASE/'runtime/ACCOUNTING.json');history=BASE/'bottlenecks/history';number=len(list(history.glob('*.json')))+1
    best=sorted(full,key=lambda r:(-r['mean_delta'],r['candidate']))
    record={'created_utc':now,'status':'REVIEWED','review_reason':args.reason,'rows':rows,'budget':assert_budget(),
       'observed_productive_research_seconds':accounting['productive_research_seconds'],
       'best_measured_full':{k:best[0][k] for k in ('candidate','mean','mean_delta')} if best else None,
       'best_second_gap':best[0]['mean_delta']-best[1]['mean_delta'] if len(best)>1 else None,
       'winner_gap_below_0_0007_is_uncertain':True,'shadow_opened':read_json(BASE/'SESSION.json').get('shadow_opened',False),
       'saturation':saturation,'finite_preregistered_menu_complete':finite_menu_complete,'failed_or_quarantined_jobs':[r['id'] for r in failures],
       'failure_interpretation':'PF003 missing -B invocation failed before model inference. Its recovery and superseded unstarted queue remain preserved; preflight passed.',
       'next_review_after_minutes':45,'decision':'Complete unchanged registered finite menus and fixed uncertainty; no parameter or seed changes from observed scores.',
       'interpretation':'Priority impacts remain subjective. Measured gains are separately recorded and are not selection-adjusted.',
       'source_sha256':digest(Path(__file__))}
    atomic_json(history/(f'{number:03d}_REVIEW.json'),record,immutable=True)
    atomic_json(BASE/'bottlenecks/BOARD.json',record);atomic_json(BASE/'bottlenecks/CURRENT.json',record)
    all_history=[read_json(p) for p in sorted(history.glob('*.json'))]
    atomic_json(ROOT/'BOTTLENECK_BOARD_HISTORY_NEXT5.json',{'updated_utc':now,'reviews':all_history})
    progress('Bottleneck review completed',args.reason+'; measured gains and uncertainty separated from subjective priorities')
    print('NEXT5 BOTTLENECK REVIEW',number,record['budget'],record['best_measured_full'],flush=True)


if __name__=='__main__':main()
