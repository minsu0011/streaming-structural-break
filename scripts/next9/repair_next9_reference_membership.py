"""Version runtime membership checks without modifying any fitted model."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now


def main():
    with activity('Repair legacy/current membership serialization with exact pinned IDs','RUNTIME_REPAIR','B9_NORMALIZATION_STATE_AVAILABILITY'):
        source=(ROOT/'src/next9/runtime_loading_v3.py').read_text(encoding='utf-8')
        old="    for model in models:assert model.training['training_ids_sha256']==scope['training_ids_sha256']"
        assert old in source
        new="""    from src.next9.runtime_membership import validate_reference_membership
    selection_path=old/'h_context_selection'/f'outer_{fold}'/'SELECTION.json'
    assert digest(selection_path)==manifest['files_sha256'][selection_path.relative_to(ROOT).as_posix()]
    selection=read_json(selection_path)
    assert selection['scope']['training_ids_sha256']==scope['training_ids_sha256']
    for model in models:validate_reference_membership(model.training,scope,selection['selection_ids'],mode)"""
        atomic(ROOT/'src/next9/runtime_loading_v4.py',source.replace(old,new))
        mapping={
            'profile_next9_m7':'profile_next9_m7_v2',
            'replay_next9_research_pool_v3':'replay_next9_research_pool_v4',
            'replay_next9_reverse_order_v3':'replay_next9_reverse_order_v4',
            'replay_next9_synthetic_portable_v3':'replay_next9_synthetic_portable_v4',
            'infer_next9_v3':'infer_next9_v4',
            'replay_next9_public_entries_v3':'replay_next9_public_entries_v4',
            'benchmark_next9_latency_v3':'benchmark_next9_latency_v4',
            'audit_next9_integrated_rp_v3':'audit_next9_integrated_rp_v4',
            'run_next9_integrated_audit_v3':'run_next9_integrated_audit_v4',
            'reproduce_next9_archive_v4':'reproduce_next9_archive_v5',
            'finalize_next9_reports_v5':'finalize_next9_reports_v6'}
        locks={f'{name}_IMPLEMENTATION_LOCK_V3_NEXT9.json':f'{name}_IMPLEMENTATION_LOCK_V4_NEXT9.json'
               for name in ['LIVE_REPLAY','REVERSE_ORDER','PORTABLE_FIXTURE','ARCHIVE_REPRODUCTION']}
        for prior,later in mapping.items():
            text=(ROOT/f'scripts/next9/{prior}.py').read_text(encoding='utf-8')
            text=text.replace('src.next9.runtime_loading_v3','src.next9.runtime_loading_v4')
            for a,b in mapping.items():
                text=text.replace(f'scripts/next9/{a}.py',f'scripts/next9/{b}.py')
                text=text.replace(f'scripts.next9.{a} import',f'scripts.next9.{b} import')
            for a,b in locks.items():text=text.replace(a,b)
            text=text.replace('DEMO_INFERENCE_V3_NEXT9.json','DEMO_INFERENCE_V4_NEXT9.json')
            compile(text,later,'exec');atomic(ROOT/f'scripts/next9/{later}.py',text)
        demo=read_json(BASE/'DEMO_INFERENCE_V3_NEXT9.json')
        demo.update(timestamp=utc_now(),source_sha256=digest(ROOT/'scripts/next9/infer_next9_v4.py'),
                    supersedes_sha256=digest(BASE/'DEMO_INFERENCE_V3_NEXT9.json'))
        atomic(BASE/'DEMO_INFERENCE_V4_NEXT9.json',demo)
        added=['src/next9/runtime_loading_v4.py','src/next9/runtime_membership.py',
               *[f'scripts/next9/{v}.py' for v in mapping.values()]]
        for a,b in locks.items():
            record=read_json(BASE/a)
            record.update(timestamp=utc_now(),supersedes_sha256=digest(BASE/a),
                source_sha256={**record['source_sha256'],**{p:digest(ROOT/p) for p in added}},
                membership_repair='Pinned exact training IDs checked under original int32 NumPy serialization for NORMAL and JSON identity for EARLY; time domain checked separately.')
            atomic(BASE/b,record)
        from src.next9.runtime_loading_v4 import load_baseline
        from src.next9.runtime_membership import validate_reference_membership
        checks=[]
        oldbase=ROOT/read_json(BASE/'SESSION_NEXT9.json')['previous_NEXT8_base']
        for fold in range(5):
            selection=read_json(oldbase/'h_context_selection'/f'outer_{fold}'/'SELECTION.json')
            for mode in ['NORMAL','EARLY']:
                baseline=load_baseline(fold,mode)
                for kind in ['j102','c205']:
                    model=getattr(baseline,kind)
                    checks.append(dict(fold=fold,kind=kind,**validate_reference_membership(model.training,selection['scope'],selection['selection_ids'],mode)))
        queue=read_json(BASE/'runtime/TASK_QUEUE.json');output=[];changes=[]
        for item in queue:
            prior=Path(item['script']).stem
            if prior not in mapping:output.append(item);continue
            if prior=='profile_next9_m7':
                assert read_json(BASE/'runtime/queue'/item['id']/'TERMINAL.json')['status']=='QUARANTINE'
            else:assert not (BASE/'runtime/queue'/item['id']).exists(),item['id']
            script=f'scripts/next9/{mapping[prior]}.py'
            replacement={**item,'id':item['id']+'_MEMBERSHIP_V4','script':script,'source_sha256':digest(ROOT/script),
                         'gates':[locks.get(g,g) for g in item['gates']]}
            output.append(replacement);changes.append(dict(original=item,replacement=replacement))
        atomic(BASE/'repairs/REFERENCE_MEMBERSHIP_SERIALIZATION_REPAIR_NEXT9.json',dict(status='PASS',timestamp=utc_now(),
            checks=checks,unchanged_model_bytes=True,no_fit_or_probability_changes=True,
            original_profile_failure_preserved='runtime/queue/PROFILE_FROZEN_M7_F0',
            changes=changes,source_sha256={p:digest(ROOT/p) for p in added}))
        atomic(BASE/'runtime/TASK_QUEUE.json',output,immutable=False)
        print(dict(status='PASS',reference_checks=len(checks),tasks_versioned=len(changes)),flush=True)


if __name__=='__main__':main()
