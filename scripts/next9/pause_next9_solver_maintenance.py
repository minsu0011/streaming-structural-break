from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,utc_now

assert not (BASE/'GLOBAL_PREDICTIONS_FROZEN_NEXT9.json').exists()
queue=read_json(BASE/'runtime/TASK_QUEUE.json')
atomic(BASE/'repairs/QUEUE_BEFORE_SOLVER_MAINTENANCE_NEXT9.json',dict(timestamp=utc_now(),tasks=queue,
    reason='Observed M2/M3 maxiter250 nonconvergence before outer release. Running owned work may finish; hold new dispatch during objective-preserving diagnosis.'))
atomic(BASE/'runtime/TASK_QUEUE.json',[{**item,'status':'PAUSED_NUMERICAL_REPAIR'} for item in queue],immutable=False)
