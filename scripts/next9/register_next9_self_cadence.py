"""Register qualified auxiliary deployment/cadence audit and final handoff gates."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,utc_now,activity
import xml.etree.ElementTree as ET


def main():
    proof=read_json(BASE/'SELF_CADENCE_QUALIFICATION_NEXT9.json');assert proof['status']=='PASS'
    for p,sha in proof['source_sha256'].items():assert digest(ROOT/p)==sha
    test=BASE/'tests/SELF_CADENCE_TEST_NEXT9.xml'
    for suite in ET.parse(test).getroot().iter('testsuite'):
        assert int(suite.attrib['failures'])==int(suite.attrib['errors'])==0
    with activity('Register qualified full-cadence auxiliary self-law audit with unchanged scientific decisions',
                  'SELF_LAW_FULL_CADENCE','B5_SELF_DYNAMICS_VS_FILTER_MEMORY'):
        tasks=[];script='scripts/next9/replay_next9_self_cadence.py'
        for fold in range(5):
            for mode in ['NORMAL','EARLY']:
                for part in range(2):
                    tasks.append(dict(id=f'SELF_FULL_CADENCE_F{fold}_{mode}_P{part}',status='READY',script=script,
                        source_sha256=digest(ROOT/script),arguments=['--fold',str(fold),'--mode',mode,'--part',str(part)],
                        gates=['SELF_CADENCE_QUALIFICATION_NEXT9.json'],phase='PRODUCTIVE_RESEARCH',
                        scientific_candidate=False,registered_before_execution=True))
        script='scripts/next9/summarize_next9_self_cadence.py'
        tasks.append(dict(id='SUMMARIZE_SELF_FULL_CADENCE',status='READY',script=script,source_sha256=digest(ROOT/script),
            arguments=[],gates=[f'self_cadence/outer_{f}/{m}/part_{p}/DONE.json' for f in range(5) for m in ['NORMAL','EARLY'] for p in range(2)],
            phase='PRODUCTIVE_RESEARCH',scientific_candidate=False,registered_before_execution=True))
        source=(ROOT/'src/next9/delivery_payload_v6.py').read_text(encoding='utf-8')
        source=source.replace("trees=['","trees=['self_cadence','self_cadence_qualification','",1)
        source=source.replace("elif tree=='repairs' and path.suffix=='.py'","elif tree=='repairs' and path.suffix in ('.py','.txt')")
        compile(source,'delivery_payload_v7','exec');atomic(ROOT/'src/next9/delivery_payload_v7.py',source)
        source=(ROOT/'scripts/next9/reproduce_next9_archive_v9.py').read_text(encoding='utf-8').replace('src.next9.delivery_payload_v6','src.next9.delivery_payload_v7')
        atomic(ROOT/'scripts/next9/reproduce_next9_archive_v10.py',source)
        source=(ROOT/'scripts/next9/admit_next9_final_audit_v7.py').read_text(encoding='utf-8')
        token="    assert read_json(BASE/'SAMPLING_PHASE_DONE_NEXT9.json')['status']=='PASS'"
        assert token in source;source=source.replace(token,token+"\n    assert read_json(BASE/'SELF_CADENCE_DONE_NEXT9.json')['status']=='PASS'")
        atomic(ROOT/'scripts/next9/admit_next9_final_audit_v8.py',source)
        source=(ROOT/'scripts/next9/finalize_next9_reports_v10.py').read_text(encoding='utf-8')
        source=source.replace("required=['","required=['SELF_CADENCE_DONE_NEXT9.json','",1).replace('reproduce_next9_archive_v9','reproduce_next9_archive_v10')
        source=source.replace('현재 공개 추론 진입점은',
            '자기 전이 보조 모델은 NORMAL·EARLY의 전체 6,406,314개 관측에서 현재 F를 받기 전에 순차 발행하여 별도 검증했다. 이 경로는 B=0인 이번 M5에 활성화하지 않았다. 전체 시점과 기존 stride8 비교는 SELF_CADENCE_FINDING_NEXT9_KO.md, 표본 위상 진단은 SAMPLING_PHASE_FINDING_NEXT9_KO.md에 있다. 두 분석은 사후 진단이며 새 인증이나 튜닝에 사용하지 않았다.\n\n현재 공개 추론 진입점은')
        compile(source,'finalize_next9_reports_v11','exec');atomic(ROOT/'scripts/next9/finalize_next9_reports_v11.py',source)
        q=read_json(BASE/'runtime/TASK_QUEUE.json');assert not set(r['id'] for r in q)&set(r['id'] for r in tasks)
        atomic(BASE/'registrations/SELF_FULL_CADENCE_NEXT9.json',dict(timestamp=utc_now(),tasks=tasks,test_sha256=digest(test),
            qualification_sha256=digest(BASE/'SELF_CADENCE_QUALIFICATION_NEXT9.json'),
            fixed_analysis='All ten original outer procedures, all6400 excluded groups per mode, all observed points; all selected self laws, no current F at issuance; fixed4variants batch losses on matched phase0/all groups; EARLY evaluation remains t>=400.',
            diagnostic_only_no_new_models_branches_or_certification=True,
            added_sources_sha256={p:digest(ROOT/p) for p in ['src/next9/delivery_payload_v7.py',
                'scripts/next9/reproduce_next9_archive_v10.py','scripts/next9/admit_next9_final_audit_v8.py','scripts/next9/finalize_next9_reports_v11.py']}))
        atomic(BASE/'runtime/TASK_QUEUE.json',[*q,*tasks],immutable=False)
        current=read_json(BASE/'CURRENT_CONTROLLERS_NEXT9.json')
        current.update(timestamp=utc_now(),audit_admission='admit_next9_final_audit_v8.py',
            core_archive='reproduce_next9_archive_v10.py --tag c0001',final_reports='finalize_next9_reports_v11.py --core-tag c0001',
            final_archive='reproduce_next9_archive_v10.py --tag f0001 --final')
        atomic(BASE/'CURRENT_CONTROLLERS_NEXT9.json',current,immutable=False)
        print(dict(status='REGISTERED',tasks=len(tasks)))


if __name__=='__main__':main()
