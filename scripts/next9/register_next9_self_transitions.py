from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,read_json,atomic,digest,activity,utc_now

if __name__=='__main__':
    with activity('Freeze four self-law variants and prefix summary timing','PROTOCOL','B5_SELF_DYNAMICS_VS_FILTER_MEMORY'):
        files=['src/next9/transitions.py','src/next9/transition_inputs.py','src/next9/learning.py',
            'scripts/next9/fit_next9_self_transitions.py','scripts/next9/tests/test_transition_inputs.py',
            'src/next9/data.py','src/next9/states.py','src/next9/movement_screen.py']
        atomic(BASE/'SELF_TRANSITION_IMPLEMENTATION_LOCK_NEXT9.json',dict(status='FROZEN',timestamp=utc_now(),
            source_sha256={p:digest(ROOT/p) for p in files},tests_sha256=digest(BASE/'tests/TRANSITION_INPUT_TESTS.xml'),tests_status='PASS',
            choice='T2 if paired group logloss lowerCI>0 against bothT0/T1 in at least2of3 internal panels; otherwiseT1 if beatsT0 in at least2of3; otherwiseT0. T3 remains matched-input diagnostic, never added as a fifth graph variant.',
            early_selection='Both origin and auxiliary next-prestate timestamp<=199 in training and internal validation. Late stress never selects early law.',
            first_online_surprise='Zero at t0 because previous Online law was not issued; t>=1 uses fixed law at prestate[t-1].',
            current_F_excluded_from_all_eight_summaries=True,coefficient_update_on_validation=False,
            M5_branch='Requires separately TRAIN-provisional orientation and transition increment; otherwiseexactaliasM4',
            no_new_candidate_features=True))
        queue=read_json(BASE/'runtime/TASK_QUEUE.json');additions=[]
        script='scripts/next9/fit_next9_self_transitions.py'
        for fold in range(5):
            for inner in [None,0,1,2]:
                name=f'outer_{fold}'+('' if inner is None else f'_inner_{inner}')
                additions.append(dict(id='SELF_TRANSITIONS_'+name,status='READY',script=script,source_sha256=digest(ROOT/script),
                    arguments=['--fold',str(fold)]+([] if inner is None else ['--inner',str(inner)]),
                    gates=['TARGET_FREE_CANDIDATES_FROZEN_NEXT9.json','SELF_TRANSITION_IMPLEMENTATION_LOCK_NEXT9.json'],
                    phase='PRODUCTIVE_RESEARCH',scientific_candidate=False,registered_before_execution=True))
        assert not set(v['id'] for v in queue)&set(v['id'] for v in additions)
        atomic(BASE/'registrations/SELF_TRANSITION_TASKS.json',dict(timestamp=utc_now(),tasks=additions))
        atomic(BASE/'runtime/TASK_QUEUE.json',queue+additions,immutable=False)
