from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,utc_now

script='scripts/next9/replay_next9_self_cadence.py'
task=dict(id='QUALIFY_SELF_FULL_CADENCE',status='READY',script=script,source_sha256=digest(ROOT/script),
    arguments=['--fold','0','--mode','NORMAL','--part','0','--qualify'],gates=['SCIENTIFIC_RESULTS_NEXT9.json'],
    phase='PRODUCTIVE_RESEARCH',scientific_candidate=False,registered_before_execution=True)
q=read_json(BASE/'runtime/TASK_QUEUE.json');assert task['id'] not in [r['id'] for r in q]
atomic(BASE/'registrations/SELF_CADENCE_QUALIFICATION_NEXT9.json',dict(timestamp=utc_now(),task=task,
    source_sha256={p:digest(ROOT/p) for p in ['src/next9/self_cadence.py',script]},
    planned_followup='After numerical qualification, ten outer fitted procedures on all their original held-out groups and all observed points. Selected auxiliary laws execute before current F; fixed T0/T1/T2/T3 batch laws compare phase0 and all-point losses on identical groups.',
    scientific_scope='Auxiliary deployment and sampling diagnosis. No break labels, model training, new candidate/branch activation or promotion changes.',
    tolerances=dict(probability_absolute=1e-12,path_absolute=1e-6,path_relative=1e-7,state_and_momentum='bitwise'),
    qualification_selection='First <200 and first >=512 observation series within fold0 part0 in original order; outcome independent.'))
atomic(BASE/'runtime/TASK_QUEUE.json',[task,*q],immutable=False)
print(dict(status='REGISTERED'))
