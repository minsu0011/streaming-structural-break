"""Admit tested exact inference and carry corrected evidence into final audits."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now


def main():
    for name in ['FAST_HEADS_QUALIFICATION_NEXT9.json','SELECTED_ENGINE_QUALIFICATION_NEXT9.json']:
        r=read_json(BASE/name);assert r['status']=='PASS'
        for p,sha in r['source_sha256'].items():assert digest(ROOT/p)==sha
    with activity('Freeze exact compiled input and selected-engine inference after actual-model parity','RUNTIME_OPTIMIZATION','B9_NORMALIZATION_STATE_AVAILABILITY'):
        source=(ROOT/'src/next9/runtime_loading_v4.py').read_text(encoding='utf-8').replace('src.next9.runtime_heads_v3','src.next9.runtime_heads_v6')
        atomic(ROOT/'src/next9/runtime_loading_v5.py',source)
        mapping={
            'replay_next9_research_pool_v4':'replay_next9_research_pool_v5',
            'replay_next9_reverse_order_v4':'replay_next9_reverse_order_v5',
            'replay_next9_synthetic_portable_v4':'replay_next9_synthetic_portable_v5',
            'infer_next9_v4':'infer_next9_v5','replay_next9_public_entries_v4':'replay_next9_public_entries_v5',
            'benchmark_next9_latency_v4':'benchmark_next9_latency_v5','audit_next9_integrated_rp_v4':'audit_next9_integrated_rp_v5',
            'run_next9_integrated_audit_v4':'run_next9_integrated_audit_v5',
            'reproduce_next9_archive_v5':'reproduce_next9_archive_v6','finalize_next9_reports_v6':'finalize_next9_reports_v7',
            'profile_next9_m7_v2':'profile_next9_m7_v3'}
        locks={f'{name}_IMPLEMENTATION_LOCK_V4_NEXT9.json':f'{name}_IMPLEMENTATION_LOCK_V5_NEXT9.json'
               for name in ['LIVE_REPLAY','REVERSE_ORDER','PORTABLE_FIXTURE','ARCHIVE_REPRODUCTION']}
        for prior,later in mapping.items():
            source=(ROOT/f'scripts/next9/{prior}.py').read_text(encoding='utf-8')
            source=source.replace('src.next9.runtime_loading_v4','src.next9.runtime_loading_v5')
            source=source.replace('from src.next9.streaming import ResearchStream,IntegratedStream','from src.next9.streaming import ResearchStream\n    from src.next9.integrated_selected import IntegratedStream')
            source=source.replace('from src.next9.streaming import IntegratedStream','from src.next9.integrated_selected import IntegratedStream')
            source=source.replace('src.next9.delivery_payload_v2','src.next9.delivery_payload_v3')
            source=source.replace('DEMO_INFERENCE_V4_NEXT9.json','DEMO_INFERENCE_V5_NEXT9.json')
            for a,b in mapping.items():
                source=source.replace(f'scripts/next9/{a}.py',f'scripts/next9/{b}.py').replace(f'scripts.next9.{a} import',f'scripts.next9.{b} import')
            for a,b in locks.items():source=source.replace(a,b)
            if prior=='profile_next9_m7_v2':source=source.replace('M7_RUNTIME_PROFILE_NEXT9.json','FAST_M7_RUNTIME_PROFILE_NEXT9.json')
            if prior=='finalize_next9_reports_v6':
                source=source.replace("required=['JOINT_SOLVER_AUDIT_NEXT9.json'","required=['LOGLOSS_SIGN_REPAIR_AUDIT_NEXT9.json','FAST_HEADS_QUALIFICATION_NEXT9.json','SELECTED_ENGINE_QUALIFICATION_NEXT9.json','JOINT_SOLVER_AUDIT_NEXT9.json'")
                source=source.replace('reproduce_next9_archive_v4 --final','reproduce_next9_archive_v6 --final').replace('src/next9/runtime_loading_v3.py','src/next9/runtime_loading_v5.py')
                source=source.replace('Float32 endpoint logloss overflow recovered from frozen predictions; prior finite calculations and rank values retained. Original failure and supplemental source receipt preserved.',
                    'Unsigned-label negative logloss was invalid. All6457 supported internal panels reconstructed in float64 with the same predictions and bootstrap; candidate selection independently unchanged and reverified. Corrected outer loss applied before outer orientation execution. Legacy membership and runtime schema repairs separately recorded.')
                source=source.replace("lead+=(BASE/'SCIENTIFIC_READ_FIRST_NEXT9_KO.md')", "lead+='중요한 수치 수정: 이전 orientation_selection 내부 로그손실은 uint8 부호 오류로 무효다. orientation_logloss_corrected 자료가 최종 근거이며, 6,457개 지원 패널을 재계산해 선택 불변을 확인했다. TS-AUC·방향성·학습 모델은 변하지 않았다. LOGLOSS_SIGN_REPAIR_AUDIT_NEXT9.json에 전체 검사가 기록돼 있다.\\n\\n'\n        lead+=(BASE/'SCIENTIFIC_READ_FIRST_NEXT9_KO.md')")
            compile(source,later,'exec');atomic(ROOT/f'scripts/next9/{later}.py',source)
        source=(ROOT/'src/next9/delivery_payload_v2.py').read_text(encoding='utf-8').replace("trees=['candidates','orientation_selection'","trees=['candidates','orientation_logloss_corrected','orientation_selection'")
        atomic(ROOT/'src/next9/delivery_payload_v3.py',source)
        source=(ROOT/'scripts/next9/admit_next9_final_audit_v3.py').read_text(encoding='utf-8')
        source=source.replace("assert read_json(BASE/'JOINT_SOLVER_AUDIT_NEXT9.json')['status']=='PASS'", "assert read_json(BASE/'JOINT_SOLVER_AUDIT_NEXT9.json')['status']=='PASS'\n    assert read_json(BASE/'LOGLOSS_SIGN_REPAIR_AUDIT_NEXT9.json')['status']=='PASS'\n    assert read_json(BASE/'SELECTED_ENGINE_QUALIFICATION_NEXT9.json')['status']=='PASS'")
        atomic(ROOT/'scripts/next9/admit_next9_final_audit_v4.py',source)
        source=(ROOT/'scripts/next9/audit_next9_execution.py').read_text(encoding='utf-8')
        source=source.replace("tasks=read_json(BASE/'runtime/TASK_QUEUE.json');queue=[]", "assert read_json(BASE/'LOGLOSS_SIGN_REPAIR_AUDIT_NEXT9.json')['status']=='PASS'\n        tasks=read_json(BASE/'runtime/TASK_QUEUE.json');queue=[]")
        source=source.replace("trees=['orientation_selection'","trees=['orientation_logloss_corrected','orientation_selection'")
        source=source.replace("original_failed_queue_tasks=sum(r['status']!='COMPLETE' for r in queue),", "original_failed_queue_tasks=sum(r['status']!='COMPLETE' for r in queue),\n            additional_preserved_failed_tasks=['PROFILE_FROZEN_M7_F0','QUALIFY_FAST_HEADS_ALL10_MENU_V2'],\n            membership_and_runtime_shape_recovery_passed=True,signed_loss_repair_sha256=digest(BASE/'LOGLOSS_SIGN_REPAIR_AUDIT_NEXT9.json'),")
        atomic(ROOT/'scripts/next9/audit_next9_execution_v2.py',source)
        demo=read_json(BASE/'DEMO_INFERENCE_V4_NEXT9.json');demo.update(timestamp=utc_now(),source_sha256=digest(ROOT/'scripts/next9/infer_next9_v5.py'),
            supersedes_sha256=digest(BASE/'DEMO_INFERENCE_V4_NEXT9.json'),runtime='Exact compiled empty-rule input and selected first engine; unchanged frozen weights and fallback branches.')
        atomic(BASE/'DEMO_INFERENCE_V5_NEXT9.json',demo)
        added=['src/next9/runtime_loading_v5.py','src/next9/runtime_heads_v6.py','src/next9/fast_design_v2.py','src/next9/integrated_selected.py','src/next9/delivery_payload_v3.py',
               'scripts/next9/admit_next9_final_audit_v4.py','scripts/next9/audit_next9_execution_v2.py',*[f'scripts/next9/{v}.py' for v in mapping.values()]]
        for a,b in locks.items():
            r=read_json(BASE/a);r.update(timestamp=utc_now(),supersedes_sha256=digest(BASE/a),
                source_sha256={**r['source_sha256'],**{p:digest(ROOT/p) for p in added}},
                actual_model_parity_sha256=digest(BASE/'SELECTED_ENGINE_QUALIFICATION_NEXT9.json'),no_scientific_model_changes=True)
            atomic(BASE/b,r)
        q=read_json(BASE/'runtime/TASK_QUEUE.json');out=[];changes=[]
        for item in q:
            stem=Path(item['script']).stem
            if stem not in mapping or stem=='profile_next9_m7_v2':out.append(item);continue
            assert not (BASE/'runtime/queue'/item['id']).exists(),item['id']
            script=f'scripts/next9/{mapping[stem]}.py'
            new={**item,'id':item['id']+'_FAST_V5','script':script,'source_sha256':digest(ROOT/script),
                 'gates':[locks.get(g,g) for g in item['gates']]}
            out.append(new);changes.append(dict(original=item,replacement=new))
        assert len(changes)==51
        script='scripts/next9/profile_next9_m7_v3.py'
        profile=dict(id='PROFILE_EXACT_FAST_M7_F0',status='READY',script=script,source_sha256=digest(ROOT/script),arguments=[],
            gates=['SELECTED_ENGINE_QUALIFICATION_NEXT9.json'],phase='PRODUCTIVE_RESEARCH',scientific_candidate=False,registered_before_execution=True)
        atomic(BASE/'repairs/EXACT_RUNTIME_OPTIMIZATION_NEXT9.json',dict(status='PASS',timestamp=utc_now(),changes=changes,
            source_sha256={p:digest(ROOT/p) for p in added},actual_bitwise_qualified_procedures=10,synthetic_paths_per_procedure=3,
            original_failed_schema_task_preserved=True,no_parameters_or_candidates_changed=True,latency_gate_not_yet_measured=True))
        atomic(BASE/'registrations/EXACT_RUNTIME_OPTIMIZATION_TASKS_NEXT9.json',dict(timestamp=utc_now(),profile=profile,changes=changes))
        atomic(BASE/'runtime/TASK_QUEUE.json',[profile,*out],immutable=False)
        print(dict(status='PASS',replay_tasks_versioned=len(changes)),flush=True)


if __name__=='__main__':main()
