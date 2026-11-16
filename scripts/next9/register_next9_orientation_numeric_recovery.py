from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,utc_now

queue=read_json(BASE/'runtime/TASK_QUEUE.json')
old=next(x for x in queue if x['id']=='ORIENTATION_V2_outer_4_inner_0')
assert read_json(BASE/'runtime/queue'/old['id']/'TERMINAL.json')['status']=='QUARANTINE'
path='scripts/next9/select_next9_orientation_numeric_recovery.py'
item={**old,'id':'ORIENTATION_NUMERIC_RECOVERY_outer_4_inner_0','script':path,'source_sha256':digest(ROOT/path)}
assert not any(x['id']==item['id'] for x in queue)
atomic(BASE/'repairs/ORIENTATION_FLOAT32_LOGLOSS_RECOVERY_TESTED.json',dict(status='PASS',timestamp=utc_now(),
    regression_sha256=digest(BASE/'tests/ORIENTATION_NUMERIC_RECOVERY.xml'),task=item,
    failed_task_remains_quarantined=True,original_partial_prediction_freeze_preserved=True))
queue.insert(0,item)
atomic(BASE/'runtime/TASK_QUEUE.json',queue,immutable=False)
