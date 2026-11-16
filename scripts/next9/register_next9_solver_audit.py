from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,utc_now

script='scripts/next9/summarize_next9_solver_recovery.py'
scopes=read_json(BASE/'TARGET_FREE_CANDIDATES_FROZEN_NEXT9.json')['candidates']
task=dict(id='AUDIT_JOINT_SOLVER_CONVERGENCE',status='READY',script=script,source_sha256=digest(ROOT/script),arguments=[],
    gates=[f'joint_heads/{r["scope"]}/DONE.json' for r in scopes],phase='PRODUCTIVE_RESEARCH',scientific_candidate=False,registered_before_execution=True)
queue=read_json(BASE/'runtime/TASK_QUEUE.json');assert not any(r['id']==task['id'] for r in queue)
atomic(BASE/'registrations/SOLVER_CONVERGENCE_AUDIT_TASK.json',dict(timestamp=utc_now(),task=task,
    purpose='Numerical convergence,notvalidationqualityselection. First250iterationwarningsretained;onlyfinalsameobjectiveconvergencecounts.'))
# Put the gate immediately after joint fits, before any outer release work.
index=max(i for i,r in enumerate(queue) if r['id'].startswith('JOINT_HEADS_'))+1
queue.insert(index,task)
atomic(BASE/'runtime/TASK_QUEUE.json',queue,immutable=False)
