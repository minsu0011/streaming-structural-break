from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now


def main():
    with activity('Freeze complete RP raw streaming replay and state-only serialization checks','RUNTIME_PROTOCOL','B9_NORMALIZATION_STATE_AVAILABILITY'):
        files=['scripts/next9/replay_next9_research_pool.py','src/next9/streaming.py','src/next9/runtime_loading.py',
            'src/next9/replay_checks.py','src/next9/joint_design.py','src/next9/transitions.py','src/next9/transition_inputs.py',
            'src/next9/cross.py','src/next9/learning.py','src/next9/states.py','src/next9/meta_correction.py',
            'src/next9/boundary.py','src/next9/data.py','src/next8/features.py','src/next5/runtime.py',
            'scripts/next9/tests/test_streaming.py','scripts/next9/qualify_next9_streaming.py']
        for name in files:compile((ROOT/name).read_text(encoding='utf-8'),name,'exec')
        qualification=BASE/'tests/RAW_STREAMING_SCHEMA_QUALIFICATION_NEXT9.json'
        assert read_json(qualification)['status']=='PASS'
        for name,sha in read_json(qualification)['source_sha256'].items():assert digest(ROOT/name)==sha
        atomic(BASE/'LIVE_REPLAY_IMPLEMENTATION_LOCK_NEXT9.json',dict(status='FROZEN',timestamp=utc_now(),
            source_sha256={p:digest(ROOT/p) for p in files},tests_status='PASS',tests_sha256=digest(BASE/'tests/STREAMING_TESTS.xml'),
            raw_schema_test_sha256=digest(qualification),coverage='Every original RP6400 series in NORMAL and separately early-trained models;2partitions perouterfold/mode,20tasks. All onlinepoints consumed, late-only early quality compared aftert400.',
            parity='Every153raw-normalized scalar F exactly equals approved RP featurecache; all frozenM0..M7/TAB scalar probabilities within1e-7 of issued batchvalues. Separate TS-AUC parity gate<=1e-8; never call tolerance bitwise.',
            continuity='Serialize and restore mutable raw engines, seventeen past feature vectors, and BL80 numericstate at74points for everyseries. Reinitialize eachseries; reversed EARLY order; fresh identical17prefix withchangedandextendedunconsumedfuture. No futurelength API.',
            immutable_fit='Hash serialized fitted heads before andafter allseries ineachpart; no online label or coefficient update.',
            final_M7='Failed inner gate exactBL80 bypasses allnewstate. Successful gate computesonlyfrozenM5 and fixed-offset correction; graphM6 never substituted.',
            exact_reference='Matched excludedfold J102/C205, separatelyearlytrainedrefs, original trainingmembership verified; neverfullDEVmodel substitution.',
            no_labels_tau_or_future_input=True,large_prediction_arrays_excluded_from_handoff=True))
        queue=read_json(BASE/'runtime/TASK_QUEUE.json');tasks=[];script='scripts/next9/replay_next9_research_pool.py'
        for fold in range(5):
            for mode in ['NORMAL','EARLY']:
                for part in range(2):
                    tasks.append(dict(id=f'LIVE_RP_F{fold}_{mode}_P{part}',status='READY',script=script,source_sha256=digest(ROOT/script),
                        arguments=['--fold',str(fold),'--mode',mode,'--part',str(part)],
                        gates=['LIVE_REPLAY_IMPLEMENTATION_LOCK_NEXT9.json','GLOBAL_PREDICTIONS_FROZEN_NEXT9.json'],
                        phase='PRODUCTIVE_RESEARCH',scientific_candidate=False,registered_before_execution=True))
        assert not {v['id'] for v in queue}&{v['id'] for v in tasks}
        atomic(BASE/'registrations/LIVE_RP_REPLAY_TASKS.json',dict(timestamp=utc_now(),tasks=tasks))
        atomic(BASE/'runtime/TASK_QUEUE.json',queue+tasks,immutable=False)


if __name__=='__main__':main()
