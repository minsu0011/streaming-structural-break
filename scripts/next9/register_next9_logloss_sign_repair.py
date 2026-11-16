"""Prospectively fix outer loss and reconstruct all internal loss evidence."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now


with activity('Register complete signed-label loss repair without changing any selected model','NUMERICAL_REPAIR','B9_NORMALIZATION_STATE_AVAILABILITY'):
    diagnosis=read_json(BASE/'repairs/LOGLOSS_SIGN_INDEPENDENT_SELECTION_DIAGNOSIS_NEXT9.json')
    assert diagnosis['all_fail_without_using_logloss']
    original='scripts/next9/evaluate_next9_outer_orientation_v3.py';new='scripts/next9/evaluate_next9_outer_orientation_v4.py'
    text=(ROOT/original).read_text(encoding='utf-8')
    text=text.replace('src.next9.inference_numeric_recovery import compare','src.next9.inference_signed_v4 import compare')
    text=text.replace('OUTER_EVALUATION_IMPLEMENTATION_LOCK_V3_NEXT9.json','OUTER_EVALUATION_IMPLEMENTATION_LOCK_V4_NEXT9.json')
    atomic(ROOT/new,text)
    lock=read_json(BASE/'OUTER_EVALUATION_IMPLEMENTATION_LOCK_V3_NEXT9.json')
    added=[new,'src/next9/inference_signed_v4.py','src/next9/logloss_signed.py',
           'scripts/next9/recompute_next9_internal_logloss.py','scripts/next9/summarize_next9_logloss_repair.py']
    lock.update(timestamp=utc_now(),supersedes_sha256=digest(BASE/'OUTER_EVALUATION_IMPLEMENTATION_LOCK_V3_NEXT9.json'),
        source_sha256={**lock['source_sha256'],**{p:digest(ROOT/p) for p in added}},
        signed_logloss_repair=True,no_change_to_frozen_models_or_predictions=True,
        no_outer_orientation_evaluation_executed_before_repair=True)
    atomic(BASE/'OUTER_EVALUATION_IMPLEMENTATION_LOCK_V4_NEXT9.json',lock)
    tasks=[];gates=[]
    for fold in range(5):
        for inner in [None,0,1,2]:
            scope=f'outer_{fold}'+('' if inner is None else f'_inner_{inner}')
            script='scripts/next9/recompute_next9_internal_logloss.py'
            task=dict(id='RECOMPUTE_SIGNED_LOGLOSS_'+scope,status='READY',script=script,source_sha256=digest(ROOT/script),
                arguments=['--fold',str(fold)]+([] if inner is None else ['--inner',str(inner)]),
                gates=[f'orientation_selection/{scope}/DONE.json'],phase='PRODUCTIVE_RESEARCH',scientific_candidate=False,registered_before_execution=True)
            tasks.append(task);gates.append(f'orientation_logloss_corrected/{scope}/DONE.json')
    script='scripts/next9/summarize_next9_logloss_repair.py'
    tasks.append(dict(id='AUDIT_SIGNED_LOGLOSS_REPAIR',status='READY',script=script,source_sha256=digest(ROOT/script),arguments=[],gates=gates,
                     phase='PRODUCTIVE_RESEARCH',scientific_candidate=False,registered_before_execution=True))
    queue=read_json(BASE/'repairs/QUEUE_BEFORE_LOGLOSS_SIGN_REPAIR_NEXT9.json')['queue'];output=[];changes=[]
    for item in queue:
        if item['script']==original:
            assert not (BASE/'runtime/queue'/item['id']).exists(),item['id']
            replacement={**item,'id':item['id']+'_SIGNED_V4','script':new,'source_sha256':digest(ROOT/new),
                'gates':[g.replace('OUTER_EVALUATION_IMPLEMENTATION_LOCK_V3_NEXT9.json','OUTER_EVALUATION_IMPLEMENTATION_LOCK_V4_NEXT9.json') for g in item['gates']]}
            changes.append(dict(original=item,replacement=replacement));item=replacement
        if item['id'] in ['SUMMARIZE_SCIENTIFIC_RESULTS','WRITE_SCIENTIFIC_REPORTS_V2']:
            item={**item,'gates':[*item['gates'],'LOGLOSS_SIGN_REPAIR_AUDIT_NEXT9.json']}
        output.append(item)
    assert len(changes)==5
    # Keep the exact-input qualification first; corrected losses then precede
    # remaining scientific evaluation. Two already-active rank jobs may finish.
    ordered=[r for r in output if r['id'].startswith('QUALIFY_FAST_HEADS')]+tasks+[r for r in output if not r['id'].startswith('QUALIFY_FAST_HEADS')]
    atomic(BASE/'repairs/LOGLOSS_SIGN_REPAIR_REGISTERED_NEXT9.json',dict(status='PASS',timestamp=utc_now(),
        independent_selection_veto_sha256=digest(BASE/'repairs/LOGLOSS_SIGN_INDEPENDENT_SELECTION_DIAGNOSIS_NEXT9.json'),
        regression_sha256=digest(BASE/'tests/LOGLOSS_SIGNED_TEST.xml'),new_tasks=tasks,changes=changes,
        original_invalid_loss_artifacts_retained=True,original_rank_and_model_artifacts_unchanged=True))
    atomic(BASE/'runtime/TASK_QUEUE.json',ordered,immutable=False)
    print(dict(registered=len(tasks),outer_jobs_repaired=len(changes)),flush=True)
