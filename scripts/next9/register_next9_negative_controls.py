from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,read_json,atomic,digest,activity,utc_now

if __name__=='__main__':
    with activity('Register fixed paired raw-background and observation-shock controls','PROTOCOL','B5_SELF_DYNAMICS_VS_FILTER_MEMORY'):
        files=['src/next9/negative_controls.py','src/next9/states.py','src/next8/features.py',
            'scripts/next9/run_next9_negative_controls.py','scripts/next9/tests/test_negative_control_generator.py']
        atomic(BASE/'NEGATIVE_CONTROL_IMPLEMENTATION_LOCK_NEXT9.json',dict(status='FROZEN',timestamp=utc_now(),
            source_sha256={p:digest(ROOT/p) for p in files},tests_status='PASS',tests_sha256=digest(BASE/'tests/NEGATIVE_CONTROL_GENERATOR_TESTS.xml'),
            cases=168,backgrounds=4,interventions=7,seeds=[9101604,9101605,9101606],H_lengths=[512,2048],online_length=512,
            parameters=dict(burnin=512,AR=.8,ARCH_intercept=.1,ARCH_square_coefficient=.85,student_df=5,
                mean_shift=1.,scale_up=2.,scale_down=.5,dependence_after=-.5,observation_shock=8.,repeated_shock_interval=32),
            paired_noise='Same seed/background/H noise and H across all interventions',
            interpretation='Observation-only single/repeated shocks leave latent suffix unchanged, exposing recursive filter response. This distinguishes a computational-memory explanation; not evidence of market causality.',
            limitation='Synthetic metadata is outside predictor API. Small subgroup counts are descriptive and cannot satisfy original64group conditional certification.',
            no_real_outcome_candidate_selection=True))
        queue=read_json(BASE/'runtime/TASK_QUEUE.json');script='scripts/next9/run_next9_negative_controls.py'
        additions=[dict(id='NEGATIVE_CONTROLS_PART'+str(k),status='READY',script=script,source_sha256=digest(ROOT/script),arguments=['--part',str(k)],
            gates=['NEGATIVE_CONTROL_IMPLEMENTATION_LOCK_NEXT9.json','TARGET_FREE_CANDIDATES_FROZEN_NEXT9.json'],phase='PRODUCTIVE_RESEARCH',scientific_candidate=False,registered_before_execution=True) for k in range(4)]
        assert not set(v['id'] for v in queue)&set(v['id'] for v in additions)
        atomic(BASE/'registrations/NEGATIVE_CONTROL_TASKS.json',dict(timestamp=utc_now(),tasks=additions))
        atomic(BASE/'runtime/TASK_QUEUE.json',queue+additions,immutable=False)
