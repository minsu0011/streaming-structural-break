from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now

with activity('Register fixed synthetic portable fixtures for all10 excluded-fold procedures','PORTABLE_PROTOCOL','B9_NORMALIZATION_STATE_AVAILABILITY'):
    paths=['scripts/next9/replay_next9_synthetic_portable.py','src/next9/runtime_loading_v2.py','src/next9/runtime_laws.py','src/next9/streaming.py','src/next9/negative_controls_v2.py','scripts/next9/infer_next9.py']
    for path in paths:compile((ROOT/path).read_text(encoding='utf-8'),path,'exec')
    lock='PORTABLE_FIXTURE_IMPLEMENTATION_LOCK_NEXT9.json'
    atomic(BASE/lock,dict(status='FROZEN',timestamp=utc_now(),source_sha256={p:digest(ROOT/p) for p in paths},
        coverage='168 fixed synthetic cases x5outerfolds x2trainingmodes x512 scalar observations, all9 models',
        input='Synthetic H512/2048, online512, same preregistered backgrounds/interventions/seeds; no added fitting candidates',
        first_pass='Save/restore at74, compare dedicated M7 to joint M7, altered extended future prefix31 exact',
        final_audit='Fresh extracted payload, reverse case order uninterrupted, all probabilities bitwise against synthetic fixture; original workspace/data denied',
        fixture_is_synthetic_only=True,no_model_fitting=True,bootstrap_or_quality_inference=False))
    script=paths[0];tasks=[]
    for fold in range(5):
        for mode in ['NORMAL','EARLY']:
            tasks.append(dict(id=f'PORTABLE_FIXTURE_F{fold}_{mode}',status='READY',script=script,source_sha256=digest(ROOT/script),
                arguments=['--fold',str(fold),'--mode',mode],gates=[lock,'GLOBAL_PREDICTIONS_FROZEN_NEXT9.json'],
                phase='PRODUCTIVE_RESEARCH',scientific_candidate=False,registered_before_execution=True))
    queue=read_json(BASE/'runtime/TASK_QUEUE.json');assert not {x['id'] for x in tasks}&{x['id'] for x in queue}
    atomic(BASE/'registrations/PORTABLE_FIXTURE_TASKS_NEXT9.json',dict(timestamp=utc_now(),tasks=tasks))
    atomic(BASE/'runtime/TASK_QUEUE.json',queue+tasks,immutable=False)
