from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,utc_now

script='scripts/next9/profile_next9_m7.py'
task=dict(id='PROFILE_FROZEN_M7_F0',status='READY',script=script,source_sha256=digest(ROOT/script),arguments=[],
    gates=['cross_and_tabular/outer_0/DONE.json','meta_correction/outer_0/DONE.json'],phase='PRODUCTIVE_RESEARCH',
    scientific_candidate=False,registered_before_execution=True)
queue=read_json(BASE/'runtime/TASK_QUEUE.json');assert not any(r['id']==task['id'] for r in queue)
atomic(BASE/'registrations/FROZEN_M7_RUNTIME_PROFILE_NEXT9.json',dict(timestamp=utc_now(),task=task,
    question='Where is inference cost spent in the already-frozen M7? Any optimization must preserve fixed inputs,weightsand probabilities; no scientific reselection.',
    not_a_latency_certification=True,case_menu='Six existing Gaussian/NONE paths,3fixedseeds xH512/2048; noextra fit/data/label.'))
atomic(BASE/'runtime/TASK_QUEUE.json',[task]+queue,immutable=False)
