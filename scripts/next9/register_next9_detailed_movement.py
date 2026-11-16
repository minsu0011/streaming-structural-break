from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,read_json,atomic,digest,activity,utc_now

if __name__=='__main__':
    with activity('Register bounded D2 D3 D5 forecast controls before label phase','PROTOCOL','B5_SELF_DYNAMICS_VS_FILTER_MEMORY'):
        files=['src/next9/detailed_movement.py','scripts/next9/fit_next9_detailed_movement.py',
            'scripts/next9/tests/test_detailed_movement.py','src/next9/movement_screen.py','src/next9/data.py','src/next9/states.py','src/next8/features.py']
        atomic(BASE/'DETAILED_MOVEMENT_IMPLEMENTATION_LOCK_NEXT9.json',dict(status='FROZEN',timestamp=utc_now(),
            source_sha256={p:digest(ROOT/p) for p in files},test_sha256=digest(BASE/'tests/DETAILED_MOVEMENT_TESTS.xml'),tests_status='PASS',
            methods=['D2','D3','D5'],D4_alias='D3_NO_CERTIFIED_STABLE_CONTEXT',
            exact_duplicate_pruning='TRAIN-only across fixed D5 mechanical controls and ownhistory, no artificial ridge gain from repeated columns',
            candidate_freeze_remains_based_on_original_D1_only=True,outer_quality_released=False))
        queue=read_json(BASE/'runtime/TASK_QUEUE.json');additions=[];gates=[]
        script='scripts/next9/fit_next9_detailed_movement.py'
        for fold in range(5):
            for inner in [None,0,1,2]:
                name=f'outer_{fold}'+('' if inner is None else f'_inner_{inner}')
                additions.append(dict(id='DETAILED_MOVEMENT_'+name,status='READY',script=script,source_sha256=digest(ROOT/script),
                    arguments=['--fold',str(fold)]+([] if inner is None else ['--inner',str(inner)]),
                    gates=['TARGET_FREE_CANDIDATES_FROZEN_NEXT9.json','DETAILED_MOVEMENT_IMPLEMENTATION_LOCK_NEXT9.json'],
                    phase='PRODUCTIVE_RESEARCH',scientific_candidate=False,registered_before_execution=True))
                gates.append('detailed_movement/'+name+'/DONE.json')
        assert not set(v['id'] for v in queue)&set(v['id'] for v in additions)
        index=next(k for k,v in enumerate(queue) if v['id']=='OPEN_SCOPED_LABEL_GATE')
        queue[index]['gates']+=gates
        atomic(BASE/'registrations/DETAILED_MOVEMENT_TASKS.json',dict(timestamp=utc_now(),tasks=additions,
            label_gate_additional_dependencies=gates,reason='Finish detailed target-free controls before entering break-label phase; no score-dependent ordering'))
        atomic(BASE/'runtime/TASK_QUEUE.json',queue[:index]+additions+queue[index:],immutable=False)
