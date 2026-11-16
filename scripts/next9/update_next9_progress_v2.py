"""Read completed results and update progress; monitoring earns no time credit."""
from pathlib import Path
import sys
import argparse
from collections import Counter
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,utc_now

def main(review):
    from src.next9.work_time import snapshot
    heartbeat=read_json(BASE/'runtime/HEARTBEAT_NEXT9.json');clock=snapshot()
    orientation=[];rules=[]
    orientation_root=BASE/('orientation_logloss_corrected' if (BASE/'LOGLOSS_SIGN_REPAIR_AUDIT_NEXT9.json').exists() else 'orientation_selection')
    for path in orientation_root.glob('*/DONE.json'):
        for mode in ['NORMAL','EARLY']:
            r=read_json(path.parent/mode/'SELECTION_FROZEN.json');orientation.extend(r['records']);rules.extend(r['provisional_rules'])
    support=[r for r in orientation if r['status']=='COMPLETE'];raw=0;full=0
    for r in support:
        e=r['evidence'];a=e['orientation']['first'];b=e['orientation']['second']
        if a['status']!='SUPPORTED' or b['status']!='SUPPORTED':continue
        opposite=(a['ci_low']>0 and b['ci_high']<0) or (b['ci_low']>0 and a['ci_high']<0)
        raw+=int(opposite)
        good=opposite and r['difference_Holm_p']<.05
        for ref in ['O1','STATE_ONLY']:
            c=e['contrasts'][r['method']+'__vs__'+ref]
            good=good and all(c[k]['ci_low'] is not None and c[k]['ci_low']>0 for k in ['TS_AUC','logloss_gain'])
        full+=int(good)
    selected=[]
    for path in (BASE/'self_transitions').glob('*/[NE]*/F*/DONE.json'):selected.append(read_json(path)['chosen_variant'])
    counts={tree:len(list((BASE/tree).glob('*/DONE.json'))) for tree in ['orientation_selection','self_transitions','joint_heads','outer_model_evaluation']}
    phase='FROZEN_OUTER_EVALUATION' if (BASE/'GLOBAL_PREDICTIONS_FROZEN_NEXT9.json').exists() else 'JOINT_STATE_CONDITIONED_MODELS' if counts['joint_heads'] else 'SELF_TRANSITION_NETWORK'
    science=read_json(BASE/'SCIENTIFIC_RESULTS_NEXT9.json') if (BASE/'SCIENTIFIC_RESULTS_NEXT9.json').exists() else None
    if science:phase='REAL_STREAM_REPRODUCTION_AND_FORENSICS'
    current=dict(phase=phase,bottleneck='B2_NO_REPLICATED_OPPOSITE_ORIENTATION' if not rules else 'B4_EARLY_LATE_REPLICATION',
        target_free_candidates=9,certified_variable_count=None if science is None else science['B']['certified_variable_count'],
        cross_edge_count=None if science is None else science['cross_edge_count'],inner_best=None,
        next_task='Complete preregistered fit/evaluation/replay gates; diagnose failures without changing scientific thresholds. Final audit only after36000actualresearchseconds.')
    atomic(BASE/'runtime/CURRENT_RESEARCH_STATUS.json',current,immutable=False)
    record=dict(timestamp=utc_now(),phase=phase,productive_seconds=clock['productive_seconds'],audit_seconds=clock['audit_seconds'],
        completed_procedures=counts,internal_pair_panels=len(orientation),common_supported=len(support),opposite_pattern_panels=raw,
        full_B_pass_panels=full,provisional_rules=len(rules),chosen_self_law_counts=dict(Counter(selected)),
        active_tasks=heartbeat['active_tasks'],queue_failures=heartbeat['failed_tasks'],
        repaired_queue_failures=['ORIENTATION_V2_outer_4_inner_0','PROFILE_FROZEN_M7_F0','QUALIFY_FAST_HEADS_ALL10_MENU_V2'],signed_loss_repair_complete=(BASE/'LOGLOSS_SIGN_REPAIR_AUDIT_NEXT9.json').exists(),outer_released=(BASE/'OUTER_RELEASE_NEXT9.json').exists(),
        active_bottlenecks=[current['bottleneck'],'B5_SELF_DYNAMICS_VS_FILTER_MEMORY'],
        decision='Keep all frozen candidates, state thresholds and finite model menus. Evaluate remaining procedures; use matched controls and dedicated runtime/movement-equation audits to separate science failure from software failure.',
        source_hash=digest(Path(__file__)),next_review_after_minutes=45,next_review_by_minutes=60)
    if review:
        index=len(list((BASE/'bottleneck_reviews').glob('REVIEW_*.json')));record['review_index']=index
        atomic(BASE/'bottleneck_reviews'/f'REVIEW_{index:03d}.json',record)
    print(record,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--review',action='store_true');main(p.parse_args().review)
