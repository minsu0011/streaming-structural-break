"""Prospective complete outer release and inference registration."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now


def main():
    assert not (BASE/'GLOBAL_PREDICTIONS_FROZEN_NEXT9.json').exists()
    with activity('Register complete A B C outer release, exact-parent controls, and interpretation gates','PROTOCOL','B1_COMMON_CURRENT_RANGE'):
        cross_files=['scripts/next9/refit_next9_outer_cross_controls.py','src/next9/cross.py','src/next9/detailed_movement.py',
            'src/next9/data.py','src/next9/states.py','src/next9/learning.py','src/next9/labels.py','src/next9/boundary.py']
        atomic(BASE/'OUTER_CROSS_CONTROLS_LOCK_NEXT9.json',dict(status='FROZEN',timestamp=utc_now(),
            source_sha256={p:digest(ROOT/p) for p in cross_files},policy='For every TRAIN-selected edge freeze full joint law versus exact remove-one-parent law on same TRAIN origins. Retain D3/D5/single-parent forecasts for all3h. Zero edges means zero fits.',
            before_outer_release=True,no_new_proposals=True,parameters='Same fixed T2 law and C=.1; early auxiliary target<=199'))
        files=['scripts/next9/freeze_next9_outer_predictions.py','scripts/next9/evaluate_next9_outer_models.py',
            'scripts/next9/evaluate_next9_outer_orientation.py','scripts/next9/evaluate_next9_outer_movement.py',
            'scripts/next9/evaluate_next9_outer_cross.py','src/next9/certification.py','src/next9/domains.py',
            'src/next9/inference.py','src/next9/labels.py','src/next9/movement_screen.py','src/next9/eligibility.py',
            'src/next9/data.py','src/next9/states.py','src/next9/orientation_contract.py','src/next9/boundary.py',
            'scripts/next9/tests/test_certification.py','scripts/next9/tests/test_domains.py','src/next8/paired_bootstrap.py']
        for name in files:compile((ROOT/name).read_text(encoding='utf-8'),name,'exec')
        atomic(BASE/'OUTER_EVALUATION_IMPLEMENTATION_LOCK_NEXT9.json',dict(status='FROZEN',timestamp=utc_now(),
            source_sha256={p:digest(ROOT/p) for p in files},tests_status='PASS',
            tests_sha256=digest(BASE/'tests/OUTER_INTERPRETATION_TESTS.xml'),
            closure='All5 outer folds, NORMAL and EARLY: M0..M7+TAB probabilities, all O comparator predictions/ranges, D1/D2/D3/D5 forecasts, exact remove-parent predictions before break-label scores.',
            A='All3h, frozen D0A persistence/D0B Hmedian/D0C TRAIN smoothing versus D1/D2/D3/D5; D4 exactD3 alias. Whole-original-group MSE/class/reversal intervals and every fixedtime/Hdomain. No breaklabels.',
            B_ALL='Every frozen A feature and8 fixedpairs; unsupported p1. Holm family feature x8 separately eachfold/mode. Same-time raw F opposite CI signs plus adjusted difference; paired TS-AUC AND logloss improvement versus O1 and matched STATE_ONLY.',
            B_domains='All21domains support recorded for every pair. Inferential replication only for this fold TRAIN-provisional pair. Non-ALL Holm family includes every candidate x8pairs x20domains, untested p1. No post-outer subgroup search.',
            B_certificate='Same TRAIN-provisional feature/pair/pattern selected and passes B_ALL in>=4of5 normalouter folds; same pattern passes>=4of5 separately TRAIN-early-selected outerlate folds. Atleast2TIME and2nonUNKNOWN Hdomains each pass in>=2different normalouter folds with no supported opposite-pattern fold. Max6 deterministic mean frozen A rank then featureID; no other outer-based choice.',
            B_signal_role='B certificate plus O6 matched paired noninferiority lowerCI>=-.001 in>=4normalouter folds and leakage PASS. Explicit-model superiority reported only for lowerCI>0; nonsignificance is not equivalence.',
            C='Original5fold equalmean exact TS-AUC, fullpoints,2000 paired wholeoriginalgroup bootstrap. M7vsM0 forpromotion; M2vsM1,M3vsM2,M4vsM3,M4vsM2,M5vsM4,M6vsM5,M6vsTAB for mechanisms. Missing pair-support remains undefined, never .5.',
            cross_certificate='Only TRAIN-selected edges, same preselected horizon passes paired D3 andD5 forecast lowerCI>0 plus exact remove-one-parent next-state logloss lowerCI>0 and>=64groups in>=4of5normalAND>=4of5earlylatefolds. All3h reported; no horizon changed from outer. No causal claim.',
            major_collapse='A supported major TIME/H domain has>=64positive and>=64negative originalgroups. Require M7-M0 equalmean among supported folds>=-.001 and worst supported fold>=-.005; require at least2supported folds to aggregate. Any failing supported domain blocks promotion. EARLY ALL additionally requires mean>=-.001. UNKNOWN domain is reported but never counted as H replication.',
            promotion='Original fixed mean>=.0012,pairedlowerCI>0,>=4positivefolds,worst>=-.001 plus above major-domain/early gate and causality/runtime/reproduction PASS. No role changes merely from score.',
            early_H='H mappings remain TRAIN-only and time-invariant withinseries; t<=199 onlinefit, t>=400 differentseriesvalidation.',
            no_outer_candidate_state_rule_parameter_or_weight_changes=True,adaptive_RP_not_hidden_confirmation=True))
        queue=read_json(BASE/'runtime/TASK_QUEUE.json');tasks=[]
        def add(taskid,script,args,gates):
            tasks.append(dict(id=taskid,status='READY',script=script,source_sha256=digest(ROOT/script),arguments=args,
                gates=gates,phase='PRODUCTIVE_RESEARCH',scientific_candidate=False,registered_before_execution=True))
        for fold in range(5):
            add(f'OUTER_CROSS_CONTROLS_F{fold}','scripts/next9/refit_next9_outer_cross_controls.py',['--fold',str(fold)],
                ['OUTER_CROSS_CONTROLS_LOCK_NEXT9.json',f'cross_and_tabular/outer_{fold}/DONE.json'])
        gates=['OUTER_EVALUATION_IMPLEMENTATION_LOCK_NEXT9.json']
        for fold in range(5):
            for stage in ['joint_heads','meta_correction','cross_and_tabular','outer_orientation_refits','outer_movement_refits','outer_cross_controls']:
                gates.append(f'{stage}/outer_{fold}/DONE.json')
        add('GLOBAL_OUTER_PREDICTIONS_FREEZE','scripts/next9/freeze_next9_outer_predictions.py',[],gates)
        for kind in ['models','orientation','movement','cross']:
            for fold in range(5):
                add(f'OUTER_EVALUATE_{kind.upper()}_F{fold}',f'scripts/next9/evaluate_next9_outer_{kind}.py',['--fold',str(fold)],
                    ['OUTER_EVALUATION_IMPLEMENTATION_LOCK_NEXT9.json','GLOBAL_PREDICTIONS_FROZEN_NEXT9.json','OUTER_RELEASE_NEXT9.json'])
        assert not {v['id'] for v in queue}&{v['id'] for v in tasks}
        atomic(BASE/'registrations/COMPLETE_OUTER_EVALUATION_TASKS.json',dict(timestamp=utc_now(),tasks=tasks))
        atomic(BASE/'runtime/TASK_QUEUE.json',queue+tasks,immutable=False)


if __name__=='__main__':main()
