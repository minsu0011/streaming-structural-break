"""Prospective p-value resolution repair; no outer execution has occurred."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now


def main():
    assert not (BASE/'GLOBAL_PREDICTIONS_FROZEN_NEXT9.json').exists()
    assert not (BASE/'outer_orientation_evaluation').exists()
    with activity('Prospectively repair impossible domain Holm resolution while retaining full per-domain candidate pair families','PROTOCOL_REPAIR','B1_COMMON_CURRENT_RANGE'):
        old=BASE/'OUTER_EVALUATION_IMPLEMENTATION_LOCK_NEXT9.json';prior=read_json(old)
        files=['scripts/next9/evaluate_next9_outer_orientation_v2.py','scripts/next9/freeze_next9_outer_predictions_v2.py',
            'scripts/next9/tests/test_domain_multiplicity_resolution.py']
        for p in files:compile((ROOT/p).read_text(encoding='utf-8'),p,'exec')
        updated={**prior,'timestamp':utc_now(),'supersedes_sha256':digest(old),
            'source_sha256':{**prior['source_sha256'],**{p:digest(ROOT/p) for p in files}},
            'B_domains':'All21domains support for every pair; inferential replication only for this fold TRAIN-provisional pair. Holm separately WITHIN EACH prespecified domain across every frozen A candidate x8pairs; untested p1. Domains are separate replication families, not one simultaneous confidence family; retain all results and contrary-domain evidence.',
            'resolution':'2000draws => minimum centered-bootstrap p=1/2001. A per-domain family at maximum12*8=96 can attain adjusted .04798. Combining9*8*20 would impose minimum adjusted .71964 and could never pass. Static mathematical detection before any outer fit evaluation, not outcome-based relaxation.',
            'resolution_test_sha256':digest(BASE/'tests/DOMAIN_MULTIPLICITY_RESOLUTION_TEST.xml')}
        atomic(BASE/'OUTER_EVALUATION_IMPLEMENTATION_LOCK_V2_NEXT9.json',updated)
        queue=read_json(BASE/'runtime/TASK_QUEUE.json');replacements=[];result=[]
        for item in queue:
            replace=item['id']=='GLOBAL_OUTER_PREDICTIONS_FREEZE' or item['id'].startswith('OUTER_EVALUATE_')
            if not replace:result.append(item);continue
            assert not (BASE/'runtime/queue'/item['id']).exists(),item['id']
            new={**item,'id':item['id']+'_V2','gates':[g.replace('OUTER_EVALUATION_IMPLEMENTATION_LOCK_NEXT9.json','OUTER_EVALUATION_IMPLEMENTATION_LOCK_V2_NEXT9.json') for g in item['gates']]}
            if item['id']=='GLOBAL_OUTER_PREDICTIONS_FREEZE':new['script']='scripts/next9/freeze_next9_outer_predictions_v2.py'
            elif item['id'].startswith('OUTER_EVALUATE_ORIENTATION_'):new['script']='scripts/next9/evaluate_next9_outer_orientation_v2.py'
            new['source_sha256']=digest(ROOT/new['script']);result.append(new);replacements.append(dict(original=item,replacement=new))
        assert len(replacements)==21
        atomic(BASE/'repairs/DOMAIN_MULTIPLICITY_RESOLUTION_REPAIR_NEXT9.json',dict(status='PASS',timestamp=utc_now(),
            before_any_outer_evaluation=True,original_sources_and_locks_preserved=True,original_queued_tasks_never_executed=True,
            old_lock_sha256=digest(old),new_lock_sha256=digest(BASE/'OUTER_EVALUATION_IMPLEMENTATION_LOCK_V2_NEXT9.json'),
            reason=updated['resolution'],candidate_and_model_selection_unchanged=True,correction='Each domain retains full candidate x8pair Holm; separate replication families explicitly disclosed.',
            replacements=replacements))
        atomic(BASE/'registrations/COMPLETE_OUTER_EVALUATION_TASKS_V2.json',dict(timestamp=utc_now(),tasks=[r['replacement'] for r in replacements]))
        atomic(BASE/'runtime/TASK_QUEUE.json',result,immutable=False)


if __name__=='__main__':main()
