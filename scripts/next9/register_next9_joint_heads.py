from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,read_json,atomic,digest,activity,utc_now

if __name__=='__main__':
    with activity('Freeze M1 toM5 hierarchy and full heldout prediction procedure','PROTOCOL','B6_BL80_REDUNDANCY'):
        files=['src/next9/joint_design.py','src/next9/learning.py','src/next9/labels.py',
            'src/next9/transition_inputs.py','src/next9/transitions.py','src/next9/orientation_contract.py',
            'src/next9/data.py','src/next9/states.py','scripts/next9/fit_next9_joint_heads.py',
            'scripts/next9/tests/test_joint_design.py']
        atomic(BASE/'JOINT_HEAD_IMPLEMENTATION_LOCK_NEXT9.json',dict(status='FROZEN',timestamp=utc_now(),
            source_sha256={p:digest(ROOT/p) for p in files},tests_status='PASS',tests_sha256=digest(BASE/'tests/JOINT_DESIGN_TESTS.xml'),
            specification='M1 fixedF nonlinear; M2 plus9ownpast coordinates; M3 plus all4 frozen state-family main effects; M4 plus atmost6 TRAIN-provisional pair F interactions inside TRAINcommonrange; M5 plus useful self-law8summaries only for those rules.',
            exact_alias='If Bempty retain M1/M2/M3 from eligibleA features and aliasM4/M5=M3. If Aempty allheadsintercept prior. Never call empty selection a model capacity failure.',
            no_H_time_extra_inputs_in_M_menu=True,unknown_contribution='Zero conditionalF outside frozen supported pair/range, UNKNOWN. Main effects reported separately.',
            final_M_coefficients='Joint TRAIN logistic coefficients; report their actual pattern separately from standalone orientation evidence. A positive model delta alone does not certify B.',
            prediction='Training stride8; all validation points issued in batches32768, including early inner<=199 qualification and outer>=400 stress, with no validation labels read.',
            M6_M7_and_matched_tabular='Separate qualified cross/integration stage, no premature cross-edge assumption or outer evaluation'))
        queue=read_json(BASE/'runtime/TASK_QUEUE.json');additions=[]
        script='scripts/next9/fit_next9_joint_heads.py'
        for fold in range(5):
            for inner in [None,0,1,2]:
                name=f'outer_{fold}'+('' if inner is None else f'_inner_{inner}')
                additions.append(dict(id='JOINT_HEADS_'+name,status='READY',script=script,source_sha256=digest(ROOT/script),
                    arguments=['--fold',str(fold)]+([] if inner is None else ['--inner',str(inner)]),
                    gates=['JOINT_HEAD_IMPLEMENTATION_LOCK_NEXT9.json','orientation_selection/'+name+'/DONE.json','self_transitions/'+name+'/DONE.json'],
                    phase='PRODUCTIVE_RESEARCH',scientific_candidate=False,registered_before_execution=True))
        assert not set(v['id'] for v in queue)&set(v['id'] for v in additions)
        atomic(BASE/'registrations/JOINT_HEAD_TASKS.json',dict(timestamp=utc_now(),tasks=additions))
        atomic(BASE/'runtime/TASK_QUEUE.json',queue+additions,immutable=False)
