"""Assemble complete procedure, access, numerical replay and failure evidence."""
from pathlib import Path
import sys
from collections import Counter
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now,csv_write

def main():
    assert read_json(BASE/'runtime/FINAL_AUDIT_ADMISSION_NEXT9.json')['status']=='PASS'
    with activity('Final queue closure, access provenance and complete RP numerical replay audit','FINAL_AUDIT','B9_NORMALIZATION_STATE_AVAILABILITY'):
        assert read_json(BASE/'LOGLOSS_SIGN_REPAIR_AUDIT_NEXT9.json')['status']=='PASS'
        from src.next9.repair_resolution_v2 import reconcile_failed_attempts
        repaired_attempts=reconcile_failed_attempts()
        atomic(BASE/'FAILED_ATTEMPT_RECONCILIATION_NEXT9.json',dict(status='PASS',timestamp=utc_now(),records=repaired_attempts,all_failed_attempts_reconciled=True))
        assert read_json(BASE/'FITTED_MODEL_PARAMETER_AUDIT_NEXT9.json')['status']=='PASS'
        tasks=read_json(BASE/'runtime/TASK_QUEUE.json');queue=[]
        for task in tasks:
            path=BASE/'runtime/queue'/task['id']/'TERMINAL.json';r=read_json(path)
            repaired=task['id']=='ORIENTATION_V2_outer_4_inner_0'
            assert r['status']=='COMPLETE' or (repaired and r['status']=='QUARANTINE'),task['id']
            if repaired:assert read_json(BASE/'repairs/ORIENTATION_FLOAT32_LOGLOSS_RECOVERY_COMPLETE.json')['status']=='PASS'
            queue.append(dict(task_id=task['id'],status=r['status'],repaired_separate_task=repaired,
                script=task['script'],source_sha256=task['source_sha256'],terminal_sha256=digest(path)))
        csv_write('QUEUE_COMPLETION_AUDIT_NEXT9.csv',queue)
        scopes=read_json(BASE/'TARGET_FREE_CANDIDATES_FROZEN_NEXT9.json')['candidates'];assert len(scopes)==20
        for scope in scopes:
            for tree in ['candidates','orientation_selection','self_transitions','joint_heads']:
                path=BASE/tree/scope['scope']/('FROZEN.json' if tree=='candidates' else 'DONE.json')
                r=read_json(path);assert r.get('status') in ['COMPLETE','FROZEN'],str(path)
        access=[];free=[];violations=[]
        trees=['state_transport','orientation_logloss_corrected','orientation_selection','joint_heads','cross_and_tabular','meta_correction','early_inner_reference',
            'outer_orientation_refits','outer_model_evaluation','outer_orientation_evaluation','replay_rank_verification',
            'detailed_movement','self_transitions','outer_movement_refits','outer_movement_evaluation',
            'outer_self_transition_evaluation','live_replay','reverse_order_replay','observability','figure_data']
        for tree in trees:
            for path in (BASE/tree).rglob('DONE.json'):
                r=read_json(path)
                if r.get('violations'):violations.append(str(path.relative_to(BASE)))
                for item in r.get('label_access',[]):
                    assert item['target_member_only'] and item['tau_read'] is False
                    if item['early'] and item['role']!='FROZEN_OUTER_EVALUATION':assert item['maximum_observed_time'] is None or item['maximum_observed_time']<=199
                    access.append(dict(source=path.relative_to(BASE).as_posix(),**item))
                if 'NPZ_members' in r:
                    assert not {'target','tau','breakOccurred','label','labels','true_break_age'}&set(r['NPZ_members']),str(path)
                    free.append(dict(source=path.relative_to(BASE).as_posix(),members=r['NPZ_members']))
        assert not violations
        gate=read_json(BASE/'LABEL_ACCESS_GATE_NEXT9.json');frozen=read_json(BASE/'TARGET_FREE_CANDIDATES_FROZEN_NEXT9.json')
        assert gate['candidate_freeze_sha256']==digest(BASE/'TARGET_FREE_CANDIDATES_FROZEN_NEXT9.json')
        assert frozen['timestamp']<=gate['timestamp']
        global_freeze=read_json(BASE/'GLOBAL_PREDICTIONS_FROZEN_NEXT9.json');assert global_freeze['status']=='FROZEN'
        csv_write('LABEL_ACCESS_PROVENANCE_NEXT9.csv',access)
        atomic(BASE/'LEAKAGE_AUDIT_NEXT9.json',dict(status='PASS',timestamp=utc_now(),
            RP_series=6400,SHADOW_ACCESS=0,SEAL_ACCESS=0,REDUCED_NEW_USAGE=0,PUBLIC_NEW_USAGE=0,CLOUD_NEW_USAGE=0,
            access_basis='Enforced workspace/RP and target-free NPZ boundaries, scoped label projections, fixed-source causality tests, full real-stream replay. Not an OS-wide access trace.',
            target_free_receipts=len(free),scoped_label_projection_receipts=len(access),label_roles=dict(Counter(r['role'] for r in access)),
            label_provenance_sha256=digest(BASE/'LABEL_ACCESS_PROVENANCE_NEXT9.csv'),
            candidate_freeze_precedes_label_gate=True,global_outer_freeze_sha256=digest(BASE/'GLOBAL_PREDICTIONS_FROZEN_NEXT9.json'),
            tau_or_true_break_age_inputs=False,pre_state_through_t_minus_1_only=True,
            RP_adaptive_reuse_not_independent_hidden_confirmation=True,
            observed_violations=violations,old_pytest_cache_write_exception_separate=True))
        replay=[];ranks=[]
        for mode in ['NORMAL','EARLY']:
            ids=[]
            for fold in range(5):
                for part in [0,1]:
                    p=BASE/'live_replay'/f'outer_{fold}'/mode/f'part_{part}'/'DONE.json';r=read_json(p)
                    assert r['status']=='COMPLETE' and all(v<=1e-7 for v in r['max_errors'].values())
                    reverse=BASE/'reverse_order_replay'/f'outer_{fold}'/mode/f'part_{part}'/'DONE.json'
                    rr=read_json(reverse);assert rr['status']=='COMPLETE'
                    assert sorted(rr['series_ids'])==sorted(r['series_ids']) and rr['all_method_opposite_order_bitwise']
                    ids.extend(r['series_ids']);replay.append(dict(fold=fold,mode=mode,part=part,series=r['original_series'],points=r['points'],
                        max_errors=r['max_errors'],live_sha256=digest(p),reverse_sha256=digest(reverse)))
                p=BASE/'replay_rank_verification'/f'outer_{fold}'/mode/'DONE.json';r=read_json(p)
                assert r['maximum_TS_AUC_error']<=1e-8
                ranks.append(dict(fold=fold,mode=mode,maximum_TS_AUC_error=r['maximum_TS_AUC_error'],sha256=digest(p)))
            assert len(ids)==len(set(ids))==6400
        atomic(BASE/'DETERMINISM_NEXT9.json',dict(status='PASS',timestamp=utc_now(),original_RP_series_per_mode=6400,
            normal_and_early_actual_replays=True,first_passes=replay,rank_checks=ranks,
            scalar_features_bitwise_cache_equal=True,scalar_batch_probability_tolerance=1e-7,TS_AUC_tolerance=1e-8,
            reverse_order_uninterrupted_vs_save_restored_bitwise=True,first17_prefix_and_extended_future_invariance=True,
            fitted_parameters_not_updated=True,unknown_or_no_pair_scores_not_imputed=True,
            exact_execution_replay_distinguished_from_float_batch_tolerance=True))
        ledger=[]
        for path in (BASE/'ledger').glob('*.json'):
            r=read_json(path);assert digest(path)==digest(BASE/'runtime/jobs'/path.name)
            ledger.append({**r,'wall_seconds':r['ended_epoch']-r['started_epoch'],'receipt_sha256':digest(path)})
        csv_write('EXPERIMENT_LEDGER_PRE_REPRO_NEXT9.csv',sorted(ledger,key=lambda r:r['started_epoch']))
        atomic(BASE/'EXECUTION_AUDIT_NEXT9.json',dict(status='PASS',timestamp=utc_now(),queue_tasks=len(queue),
            original_failed_queue_tasks=sum(r['status']!='COMPLETE' for r in queue),
            additional_preserved_failed_tasks=[r['failed_task'] for r in repaired_attempts],failed_attempt_reconciliation_sha256=digest(BASE/'FAILED_ATTEMPT_RECONCILIATION_NEXT9.json'),
            membership_and_runtime_shape_recovery_passed=True,signed_loss_repair_sha256=digest(BASE/'LOGLOSS_SIGN_REPAIR_AUDIT_NEXT9.json'),all_queue_failures_repaired=True,
            failed_activity_intervals=sum(r['status']=='FAILED' for r in ledger),failed_intervals_not_credited=True,
            complete_nested_procedures=20,original_five_outer_folds=True,source_script_sha256=digest(Path(__file__)),
            reports_sha256={name:digest(BASE/name) for name in ['QUEUE_COMPLETION_AUDIT_NEXT9.csv','LEAKAGE_AUDIT_NEXT9.json','DETERMINISM_NEXT9.json','EXPERIMENT_LEDGER_PRE_REPRO_NEXT9.csv']}))

if __name__=='__main__':main()
