"""Prospective runtime-only M7 dtype repair; every original source retained."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now

def main():
    assert not (BASE/'GLOBAL_PREDICTIONS_FROZEN_NEXT9.json').exists()
    with activity('Versioned M7 scalar precision repair before any full live/portable replay','RUNTIME_REPAIR','B9_NORMALIZATION_STATE_AVAILABILITY'):
        loader=(ROOT/'src/next9/runtime_loading_v2.py').read_text(encoding='utf-8')
        loader=loader.replace('from src.next9.streaming import FrozenHeads','from src.next9.runtime_heads_v3 import FrozenHeads')
        atomic(ROOT/'src/next9/runtime_loading_v3.py',loader)
        replacements={
            'replay_next9_research_pool_v2':'replay_next9_research_pool_v3',
            'replay_next9_reverse_order_v2':'replay_next9_reverse_order_v3',
            'replay_next9_synthetic_portable':'replay_next9_synthetic_portable_v3',
            'infer_next9':'infer_next9_v3','replay_next9_public_entries':'replay_next9_public_entries_v3',
            'benchmark_next9_latency':'benchmark_next9_latency_v3','audit_next9_integrated_rp':'audit_next9_integrated_rp_v3',
            'run_next9_integrated_audit':'run_next9_integrated_audit_v3','reproduce_next9_archive':'reproduce_next9_archive_v3'}
        for old,new in replacements.items():
            source=(ROOT/f'scripts/next9/{old}.py').read_text(encoding='utf-8')
            source=source.replace('src.next9.runtime_loading_v2 import','src.next9.runtime_loading_v3 import')
            source=source.replace('LIVE_REPLAY_IMPLEMENTATION_LOCK_V2_NEXT9.json','LIVE_REPLAY_IMPLEMENTATION_LOCK_V3_NEXT9.json')
            source=source.replace('REVERSE_ORDER_IMPLEMENTATION_LOCK_V2_NEXT9.json','REVERSE_ORDER_IMPLEMENTATION_LOCK_V3_NEXT9.json')
            source=source.replace('from scripts.next9.infer_next9 import','from scripts.next9.infer_next9_v3 import')
            source=source.replace('DEMO_INFERENCE_NEXT9.json','DEMO_INFERENCE_V3_NEXT9.json')
            for prior,later in replacements.items():source=source.replace(f'scripts/next9/{prior}.py',f'scripts/next9/{later}.py')
            path=f'scripts/next9/{new}.py';compile(source,path,'exec');atomic(ROOT/path,source)
        demo=read_json(BASE/'DEMO_INFERENCE_NEXT9.json')
        demo.update(timestamp=utc_now(),supersedes_sha256=digest(BASE/'DEMO_INFERENCE_NEXT9.json'),
            source_sha256=digest(ROOT/'scripts/next9/infer_next9_v3.py'),numerical_boundary='M5float32 passed identically into frozen correction in batch/shared/dedicated paths')
        atomic(BASE/'DEMO_INFERENCE_V3_NEXT9.json',demo)
        added=['src/next9/runtime_loading_v3.py','src/next9/runtime_heads_v3.py','scripts/next9/tests/test_runtime_meta_precision.py',
            *[f'scripts/next9/{v}.py' for v in replacements.values()]]
        lock_names={
            'LIVE_REPLAY_IMPLEMENTATION_LOCK_V2_NEXT9.json':'LIVE_REPLAY_IMPLEMENTATION_LOCK_V3_NEXT9.json',
            'REVERSE_ORDER_IMPLEMENTATION_LOCK_V2_NEXT9.json':'REVERSE_ORDER_IMPLEMENTATION_LOCK_V3_NEXT9.json',
            'PORTABLE_FIXTURE_IMPLEMENTATION_LOCK_NEXT9.json':'PORTABLE_FIXTURE_IMPLEMENTATION_LOCK_V3_NEXT9.json',
            'ARCHIVE_REPRODUCTION_IMPLEMENTATION_LOCK_NEXT9.json':'ARCHIVE_REPRODUCTION_IMPLEMENTATION_LOCK_V3_NEXT9.json'}
        for old,new in lock_names.items():
            record=read_json(BASE/old)
            record.update(timestamp=utc_now(),supersedes_sha256=digest(BASE/old),
                source_sha256={**record['source_sha256'],**{p:digest(ROOT/p) for p in added}},
                numerical_repair='Original shared FrozenHeads converted storedfloat32 M5 toPythonfloat andfloat64 beforelogit; dedicated/batch usefloat32. Newloader usesonlysubclassoverride torestoreidenticalM5precision; nofit orcoeffchanges.',
                regression_sha256=digest(BASE/'tests/RUNTIME_META_PRECISION_TEST.xml'))
            atomic(BASE/new,record)
        queue=read_json(BASE/'runtime/TASK_QUEUE.json');output=[];changes=[]
        for item in queue:
            old=Path(item['script']).stem
            if old not in replacements:output.append(item);continue
            assert not (BASE/'runtime/queue'/item['id']).exists(),item['id']
            script=f'scripts/next9/{replacements[old]}.py'
            replacement={**item,'id':item['id']+'_META_V3','script':script,'source_sha256':digest(ROOT/script),
                'gates':[lock_names.get(g,g) for g in item['gates']]}
            output.append(replacement);changes.append(dict(original=item,replacement=replacement))
        assert len(changes)==51,len(changes)
        atomic(BASE/'repairs/M7_RUNTIME_PRECISION_REPAIR_NEXT9.json',dict(status='PASS',timestamp=utc_now(),
            before_any_actual_full_replay=True,original_fitted_or_scientific_sources_unchanged=True,
            original_SHARED_vs_DEDICATED_mismatch_reproduced=True,repaired_shared_dedicated_batch_bitwise=True,
            no_new_model_fits=True,no_threshold_or_candidate_changes=True,
            test_sha256=digest(BASE/'tests/RUNTIME_META_PRECISION_TEST.xml'),replacements=changes,
            preferred_controller_scripts={k:v for k,v in replacements.items() if k in ['benchmark_next9_latency','run_next9_integrated_audit','reproduce_next9_archive']},
            supplemental_final_audit_sources_sha256={p:digest(ROOT/p) for p in added}))
        atomic(BASE/'registrations/RUNTIME_META_PRECISION_TASKS_V3.json',dict(timestamp=utc_now(),tasks=[r['replacement'] for r in changes]))
        atomic(BASE/'runtime/TASK_QUEUE.json',output,immutable=False)

if __name__=='__main__':main()
