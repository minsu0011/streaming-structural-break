from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,read_json,atomic,digest,activity,utc_now


if __name__=='__main__':
    with activity('Register finite nested orientation selection and guarded label phase','PROTOCOL','B1_COMMON_CURRENT_RANGE'):
        files=['src/next9/learning.py','src/next9/labels.py','src/next9/inference.py',
            'src/next9/transitions.py','src/next9/orientation_contract.py',
            'src/next9/data.py','src/next9/states.py','src/next9/boundary.py',
            'scripts/next9/select_next9_orientation.py','scripts/next9/open_next9_label_gate.py',
            'scripts/next9/tests/test_scoped_labels.py','scripts/next9/tests/test_learning_contract.py',
            'scripts/next9/tests/test_transition_contract.py','scripts/next9/tests/test_inference_contract.py',
            'scripts/next9/tests/test_orientation_integration.py',
            'src/next8/paired_bootstrap.py','src/scoring/ts_auc.py']
        atomic(BASE/'ORIENTATION_SELECTION_IMPLEMENTATION_LOCK_NEXT9.json',dict(status='FROZEN',timestamp=utc_now(),
            source_sha256={p:digest(ROOT/p) for p in files},
            protocol_details_sha256=digest(BASE/'ORIENTATION_DETAILS_PROTOCOL_NEXT9.json'),
            test_receipts_sha256={n:digest(BASE/'tests'/n) for n in ['SCOPED_LABEL_TESTS.xml','LEARNING_CONTRACT_TESTS.xml','TRANSITION_CONTRACT_TESTS.xml','ORIENTATION_INTEGRATION_TESTS.xml']},
            test_status='PASS',RP_one_original_dependency_group_per_series_verified=True,
            source_reuse='Exact NEXT8 PreparedRank numeric kernel/canonical scorer, NEXT9 bootstrap seed and key; no old artifact writes',
            within_panel_fit='Comparators fit on all admitted TRAIN rows; common-F pair masks frozen from TRAIN and used unchanged for heldout effect and comparator assessment',
            nuisance_controls='Two additional matched state-only nuisance fits for O3 and O4, not additional candidate families',
            outer_quality_released=False))
        queue=read_json(BASE/'runtime/TASK_QUEUE.json')
        additions=[]
        def add(task,script,args,gates):
            additions.append(dict(id=task,status='READY',script=script,source_sha256=digest(ROOT/script),arguments=args,
                gates=gates,phase='PRODUCTIVE_RESEARCH',scientific_candidate=False,registered_before_execution=True))
        add('OPEN_SCOPED_LABEL_GATE','scripts/next9/open_next9_label_gate.py',[],
            ['TARGET_FREE_CANDIDATES_FROZEN_NEXT9.json','ORIENTATION_SELECTION_IMPLEMENTATION_LOCK_NEXT9.json'])
        for fold in range(5):
            for inner in [None,0,1,2]:
                name=f'ORIENTATION_outer_{fold}'+('' if inner is None else f'_inner_{inner}')
                args=['--fold',str(fold)]+([] if inner is None else ['--inner',str(inner)])
                add(name,'scripts/next9/select_next9_orientation.py',args,
                    ['LABEL_ACCESS_GATE_NEXT9.json','ORIENTATION_SELECTION_IMPLEMENTATION_LOCK_NEXT9.json'])
        assert not set(v['id'] for v in queue)&set(v['id'] for v in additions)
        atomic(BASE/'registrations/ORIENTATION_SELECTION_TASKS.json',dict(timestamp=utc_now(),tasks=additions,
            procedures=20,modes=['NORMAL','EARLY'],candidate_features_fixed_by_A=True,scientific_menu_not_expanded=True))
        atomic(BASE/'runtime/TASK_QUEUE.json',queue+additions,immutable=False)
