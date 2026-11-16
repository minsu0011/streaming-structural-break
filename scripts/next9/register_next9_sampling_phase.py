from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,utc_now
import xml.etree.ElementTree as ET

test=BASE/'tests/SAMPLING_PHASE_TEST_NEXT9.xml'
for suite in ET.parse(test).getroot().iter('testsuite'):
    assert int(suite.attrib['failures'])==int(suite.attrib['errors'])==0
tasks=[];script='scripts/next9/analyze_next9_sampling_phase.py'
for part in range(4):
    tasks.append(dict(id=f'SAMPLING_PHASE_ALL153_P{part}',status='READY',script=script,source_sha256=digest(ROOT/script),
        arguments=['--part',str(part)],gates=['SCIENTIFIC_RESULTS_NEXT9.json'],phase='PRODUCTIVE_RESEARCH',
        scientific_candidate=False,registered_before_execution=True))
script='scripts/next9/summarize_next9_sampling_phase.py'
tasks.append(dict(id='SUMMARIZE_SAMPLING_PHASE_ALL153',status='READY',script=script,source_sha256=digest(ROOT/script),
    arguments=[],gates=[f'sampling_phase/part_{p}/DONE.json' for p in range(4)],phase='PRODUCTIVE_RESEARCH',
    scientific_candidate=False,registered_before_execution=True))
source=(ROOT/'src/next9/delivery_payload_v5.py').read_text(encoding='utf-8')
source=source.replace("trees=['","trees=['sampling_phase','",1);compile(source,'delivery_payload_v6','exec')
atomic(ROOT/'src/next9/delivery_payload_v6.py',source)
source=(ROOT/'scripts/next9/reproduce_next9_archive_v8.py').read_text(encoding='utf-8').replace('src.next9.delivery_payload_v5','src.next9.delivery_payload_v6')
atomic(ROOT/'scripts/next9/reproduce_next9_archive_v9.py',source)
source=(ROOT/'scripts/next9/admit_next9_final_audit_v6.py').read_text(encoding='utf-8')
token="    assert read_json(BASE/'INDEPENDENT_TS_AUC_AUDIT_NEXT9.json')['status']=='PASS'"
assert token in source;source=source.replace(token,token+"\n    assert read_json(BASE/'SAMPLING_PHASE_DONE_NEXT9.json')['status']=='PASS'")
atomic(ROOT/'scripts/next9/admit_next9_final_audit_v7.py',source)
source=(ROOT/'scripts/next9/finalize_next9_reports_v9.py').read_text(encoding='utf-8')
source=source.replace("required=['","required=['SAMPLING_PHASE_DONE_NEXT9.json','",1).replace('reproduce_next9_archive_v8','reproduce_next9_archive_v9')
compile(source,'finalize_next9_reports_v10','exec');atomic(ROOT/'scripts/next9/finalize_next9_reports_v10.py',source)
q=read_json(BASE/'runtime/TASK_QUEUE.json');assert not set(r['id'] for r in q)&set(r['id'] for r in tasks)
atomic(BASE/'registrations/SAMPLING_PHASE_NEXT9.json',dict(timestamp=utc_now(),tasks=tasks,test_sha256=digest(test),
    source_sha256={p:digest(ROOT/p) for p in ['src/next9/sampling_phase.py','src/next9/delivery_payload_v6.py',
        'scripts/next9/reproduce_next9_archive_v9.py','scripts/next9/admit_next9_final_audit_v7.py','scripts/next9/finalize_next9_reports_v10.py']},
    preregistered_diagnostic='All153, all6400 original groups, all observed points; phases0..7 and ALL, fixed7timebins, matched complete-phase group cohorts.',
    no_outcome_driven_models_or_sampling_changes=True))
atomic(BASE/'runtime/TASK_QUEUE.json',[*q,*tasks],immutable=False)
atomic(BASE/'CURRENT_CONTROLLERS_NEXT9.json',dict(timestamp=utc_now(),
    minimum_research_seconds=36000,minimum_separate_audit_seconds=3600,
    research_live='replay_next9_research_pool_v6.py',research_reverse='replay_next9_reverse_order_v5.py',
    synthetic_fixture='replay_next9_synthetic_portable_v6.py',public_entry='infer_next9_v5.py',
    latency='benchmark_next9_latency_v6.py',final_tests='run_next9_final_tests.py',
    audit_admission='admit_next9_final_audit_v7.py',integrity='audit_next9_integrity.py --stage FINAL',
    execution='audit_next9_execution_v4.py',integrated='run_next9_integrated_audit_v5.py',equations='run_next9_equation_audit.py',
    core_archive='reproduce_next9_archive_v9.py --tag c0001',final_reports='finalize_next9_reports_v10.py --core-tag c0001',
    final_archive='reproduce_next9_archive_v9.py --tag f0001 --final',desktop='deliver_next9_desktop.py --archive-tag f0001',
    order='All research, latency, tests; admission; integrity/execution/integrated/equations/core archive; >=3600 audit; final reports/final archive/Desktop; verify all owned processes stopped.'))
print(dict(status='REGISTERED',tasks=len(tasks)))
