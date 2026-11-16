from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,utc_now
import xml.etree.ElementTree as ET

test=BASE/'tests/PAIR_KERNEL_TEST_NEXT9.xml'
for suite in ET.parse(test).getroot().iter('testsuite'):
    assert int(suite.attrib['failures'])==int(suite.attrib['errors'])==0
tasks=[]
script='scripts/next9/audit_next9_pair_kernels.py'
for fold in range(5):
    for mode in ['NORMAL','EARLY']:
        tasks.append(dict(id=f'INDEPENDENT_PAIR_KERNEL_F{fold}_{mode}',status='READY',script=script,
            source_sha256=digest(ROOT/script),arguments=['--fold',str(fold),'--mode',mode],
            gates=['GLOBAL_PREDICTIONS_FROZEN_NEXT9.json',f'outer_model_evaluation/outer_{fold}/DONE.json'],
            phase='PRODUCTIVE_RESEARCH',scientific_candidate=False,registered_before_execution=True))
script='scripts/next9/summarize_next9_pair_kernels.py'
tasks.append(dict(id='SUMMARIZE_INDEPENDENT_PAIR_KERNELS',status='READY',script=script,source_sha256=digest(ROOT/script),
    arguments=[],gates=[f'pair_kernel_audit/outer_{f}/{m}/DONE.json' for f in range(5) for m in ['NORMAL','EARLY']],
    phase='PRODUCTIVE_RESEARCH',scientific_candidate=False,registered_before_execution=True))
q=read_json(BASE/'runtime/TASK_QUEUE.json');assert not set(r['id'] for r in q)&set(r['id'] for r in tasks)
atomic(BASE/'registrations/INDEPENDENT_PAIR_KERNEL_AUDIT_NEXT9.json',dict(timestamp=utc_now(),tasks=tasks,
    test_sha256=digest(test),implementation_sha256=digest(ROOT/'src/next9/pair_kernel_audit.py'),
    reason='Independent numerical validation after discovery of unsigned logloss error; same frozen predictions, domains and 2000 original group draws.',
    outcome_use='Validation only; no new selection, thresholds, fitted models or promotion decisions.'))
# Cache diagnosis has priority; available second worker can validate statistics.
priority=[r for r in q if r['id']=='QUALIFY_EAGER_SCHEMA_CACHE']
rest=[r for r in q if r['id']!='QUALIFY_EAGER_SCHEMA_CACHE']
atomic(BASE/'runtime/TASK_QUEUE.json',[*priority,*tasks,*rest],immutable=False)
print(dict(status='REGISTERED',tasks=len(tasks)))
