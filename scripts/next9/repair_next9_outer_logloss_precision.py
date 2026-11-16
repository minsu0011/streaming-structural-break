"""Apply tested endpoint handling prospectively to outer orientation metrics."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,utc_now,activity

with activity('Freeze endpoint-safe outer logloss and record successful internal recovery','NUMERICAL_REPAIR','B9_NORMALIZATION_STATE_AVAILABILITY'):
    assert not (BASE/'GLOBAL_PREDICTIONS_FROZEN_NEXT9.json').exists()
    failed='ORIENTATION_V2_outer_4_inner_0';recovered='ORIENTATION_NUMERIC_RECOVERY_outer_4_inner_0'
    done=BASE/'orientation_selection/outer_4_inner_0/DONE.json';assert read_json(done)['status']=='COMPLETE'
    original=BASE/'runtime/queue'/failed/'attempt_1/DONE.json'
    result=read_json(BASE/'orientation_selection/outer_4_inner_0/EARLY/SELECTION_FROZEN.json')
    atomic(BASE/'repairs/ORIENTATION_FLOAT32_LOGLOSS_RECOVERY_COMPLETE.json',dict(status='PASS',timestamp=utc_now(),
        original_failed_attempt_sha256=digest(original),recovered_done_sha256=digest(done),
        supplemental_implementation_sha256=digest(BASE/'repairs/ORIENTATION_FLOAT32_LOGLOSS_RECOVERY_REGISTERED.json'),
        recovered_early_provisional_rules=result['provisional_rules'],
        receipt_interpretation='Existing DONE implementation field identifies original model/selection contract; this mandatory supplemental receipt identifies endpoint-safe numerical recovery source.',
        failed_predictions_reused=True,completed_panels_not_recomputed=True,failed_task_remains_visible=True))
    oldlock=BASE/'OUTER_EVALUATION_IMPLEMENTATION_LOCK_V2_NEXT9.json'
    names={}
    for oldname,newname in [('evaluate_next9_outer_orientation_v2','evaluate_next9_outer_orientation_v3'),('freeze_next9_outer_predictions_v2','freeze_next9_outer_predictions_v3')]:
        source=(ROOT/f'scripts/next9/{oldname}.py').read_text(encoding='utf-8')
        source=source.replace('OUTER_EVALUATION_IMPLEMENTATION_LOCK_V2_NEXT9.json','OUTER_EVALUATION_IMPLEMENTATION_LOCK_V3_NEXT9.json')
        source=source.replace('from src.next9.inference import compare','from src.next9.inference_numeric_recovery import compare')
        path=f'scripts/next9/{newname}.py';compile(source,path,'exec');atomic(ROOT/path,source);names[oldname]=path
    lock=read_json(oldlock);paths=[*names.values(),'src/next9/inference_numeric_recovery.py']
    lock.update(timestamp=utc_now(),supersedes_sha256=digest(oldlock),
        source_sha256={**lock['source_sha256'],**{p:digest(ROOT/p) for p in paths}},
        probability_endpoint_repair='Float32 clipped1 vectors usefloat64 forlogloss only; all finite previous calculations and allrank inputs unchanged; regressionPASS; beforeouterevaluation')
    atomic(BASE/'OUTER_EVALUATION_IMPLEMENTATION_LOCK_V3_NEXT9.json',lock)
    queue=read_json(BASE/'runtime/TASK_QUEUE.json');output=[];replacements=[]
    for item in queue:
        stem=Path(item['script']).stem
        if stem not in names:output.append(item);continue
        assert not (BASE/'runtime/queue'/item['id']).exists()
        new={**item,'id':item['id']+'_PRECISION_V3','script':names[stem],'source_sha256':digest(ROOT/names[stem]),
            'gates':[g.replace('OUTER_EVALUATION_IMPLEMENTATION_LOCK_V2_NEXT9.json','OUTER_EVALUATION_IMPLEMENTATION_LOCK_V3_NEXT9.json') for g in item['gates']]}
        output.append(new);replacements.append(dict(original=item,replacement=new))
    assert len(replacements)==6
    atomic(BASE/'registrations/OUTER_PRECISION_REPAIR_TASKS_V3.json',dict(timestamp=utc_now(),replacements=replacements))
    atomic(BASE/'runtime/TASK_QUEUE.json',output,immutable=False)
