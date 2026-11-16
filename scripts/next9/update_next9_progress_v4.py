"""Concise evidence-backed monitoring; never creates credited work intervals."""
from pathlib import Path
import sys,argparse
from collections import Counter
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,utc_now


def main(review):
    from src.next9.work_time import snapshot
    from src.next9.repair_resolution_v2 import reconcile_failed_attempts
    h=read_json(BASE/'runtime/HEARTBEAT_NEXT9.json');clock=snapshot()
    science=read_json(BASE/'SCIENTIFIC_RESULTS_NEXT9.json')
    orientation=read_json(BASE/'ORIENTATION_FAILURE_FINDING_NEXT9.json')
    try:repairs=reconcile_failed_attempts();repair_status='PASS';repair_error=None
    except (AssertionError,FileNotFoundError) as e:repairs=[];repair_status='PENDING_DIAGNOSIS';repair_error=str(e)
    progress={}
    for mode in ['NORMAL','EARLY']:
        progress[mode]=dict(live_series=len(list((BASE/'live_replay').glob(f'outer_*/{mode}/part_*/S*/DONE.json'))),
            reverse_series=len(list((BASE/'reverse_order_replay').glob(f'outer_*/{mode}/part_*/S*.json'))),
            target_series=6400)
    self_batches=list((BASE/'self_cadence').glob('outer_*/*/part_*/b*/DONE.json'))
    self_points=0;self_series=0
    for p in self_batches:
        r=read_json(p);self_series+=len(r['records']);self_points+=sum(v['points'] for v in r['records'])
    q=read_json(BASE/'runtime/TASK_QUEUE.json');states=Counter()
    for r in q:
        p=BASE/'runtime/queue'/r['id']/'TERMINAL.json'
        states[read_json(p)['status'] if p.exists() else 'PENDING_OR_ACTIVE']+=1
    auxiliary={name:read_json(BASE/name)['status'] if (BASE/name).exists() else 'PENDING' for name in
        ['INDEPENDENT_TS_AUC_AUDIT_NEXT9.json','STATE_OBSERVABILITY_NEXT9.json','SAMPLING_PHASE_DONE_NEXT9.json',
         'SELF_CADENCE_DONE_NEXT9.json','SCIENTIFIC_INTERPRETATION_REVIEW_NEXT9.json','LATENCY_NEXT9.json']}
    current=dict(phase='REAL_STREAM_REPRODUCTION_AND_FORENSICS',bottleneck='B2_NO_REPLICATED_OPPOSITE_ORIENTATION',
        target_free_candidates=9,certified_variable_count=science['B']['certified_variable_count'],cross_edge_count=science['cross_edge_count'],
        inner_best=None,next_task='Complete registered real-stream, full-cadence auxiliary and sampling diagnostics. Keep BL80; no model changes. Final audit after all research and >=36000 productive seconds.')
    atomic(BASE/'runtime/CURRENT_RESEARCH_STATUS.json',current,immutable=False)
    record=dict(timestamp=utc_now(),productive_seconds=clock['productive_seconds'],audit_seconds=clock['audit_seconds'],
        phase=current['phase'],active_tasks=h['active_tasks'],queue_states=dict(states),
        internal_pair_panels=orientation['panels'],common_supported=orientation['supported'],
        raw_opposite_cases=orientation['raw_opposite_cases'],all_fixed_B_gate_pass_cases=orientation['all_fixed_gate_pass_cases'],
        certified_variable_count=0,cross_edge_count=0,
        active_bottlenecks=['B2_NO_REPLICATED_OPPOSITE_ORIENTATION','B4_EARLY_LATE_REPLICATION','B5_SELF_DYNAMICS_VS_FILTER_MEMORY'],
        decision='Keep fixed scientific candidates/weights/thresholds and canonical BL80. Complete registered diagnostics and deployment/reproduction checks. No retuning or forced edges.',
        frozen_M7_score=science['C']['M7_vs_BL80'],runtime_progress=progress,auxiliary_status=auxiliary,
        self_cadence_series=self_series,self_cadence_stream_points=self_points,
        failed_attempt_reconciliation_status=repair_status,preserved_reconciled_failed_attempts=len(repairs),
        failed_attempt_names=[r['failed_task'] for r in repairs],repair_error=repair_error,
        RAM_percent=h['RAM_percent'],disk_free_bytes=h['disk_free_bytes'],
        next_review_after_minutes=45,next_review_by_minutes=60,monitoring_not_productive_credit=True,source_sha256=digest(__file__))
    if review:
        index=len(list((BASE/'bottleneck_reviews').glob('REVIEW_*.json')));record['review_index']=index
        atomic(BASE/'bottleneck_reviews'/f'REVIEW_{index:03d}.json',record)
    atomic(BASE/'runtime/PROGRESS_SUMMARY_NEXT9.json',record,immutable=False)
    print({k:record[k] for k in ['timestamp','productive_seconds','audit_seconds','active_tasks','queue_states','runtime_progress',
        'auxiliary_status','self_cadence_series','self_cadence_stream_points','preserved_reconciled_failed_attempts']},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--review',action='store_true');main(p.parse_args().review)
