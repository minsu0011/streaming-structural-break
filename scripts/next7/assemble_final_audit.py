"""Consolidate completed final checks without promoting a negative model."""
from pathlib import Path
import csv
import importlib.metadata as metadata
import platform
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.next7.common import BASE,read_json,save,csv_write,digest,utc_now,union_seconds,activity,note


def rows(path):
    with path.open(newline='',encoding='utf-8-sig') as handle:return list(csv.DictReader(handle))


def main():
    from scripts.next7.benchmark_runtime import audit_gate
    audit_gate()
    state=read_json(BASE/'runtime/SCHEDULER_STATE.json')
    assert time.time()-state['last_epoch']<90
    session=read_json(BASE/'SESSION_NEXT7.json')
    productive,audit=union_seconds(state['intervals']),union_seconds(state['audit_intervals'])
    assert productive>=session['research_minimum_seconds']==36000
    # The user made36000 productive seconds mandatory, but described audit as approximately60 minutes.
    # Completed evidence below determines audit acceptance; never pad or repeat work for the target.
    assert audit>0 and session['final_audit_target_seconds']==3600
    paths={'duration_policy':'audit/FINAL_AUDIT_DURATION_INTERPRETATION_001.json',
        'data':'audit/FINAL_DATA_BOUNDARY.json','online':'audit/FINAL_ONLINE_CONTRACTS.json',
        'training':'audit/FINAL_TRAINING_LINEAGE.json','regression':'audit/FINAL_REGRESSION_LATEST.json',
        'authority':'audit/PROTECTED_AUTHORITY_FINAL_001.json','failures':'audit/failure_registry/final_001/DONE.json',
        'runtime':'LATENCY_NEXT7_DONE.json','reproduction':'reproduction/research_002/DONE.json',
        'historical_reference_reproduction':'audit/FROZEN_REFERENCE_ISOLATED_REPLAY_001.json',
        'fresh_block_requirement':'FRESH_BLOCK_REQUIREMENT_AUDIT_NEXT7.json',
        'viewer':'visuals/v4/BROWSER_REVIEW.json','viewer_visual':'audit/VIEWER_V4_VISUAL_REVIEW_001.json'}
    evidence={key:read_json(BASE/name) for key,name in paths.items()}
    for key,result in evidence.items():
        assert result['status'] in ('PASS','COMPLETE'),key
    assert evidence['regression']['failures']==evidence['regression']['errors']==0
    assert not evidence['failures']['unresolved_engineering_failures']
    assert evidence['authority']['protected_files']==6333 and not evidence['authority']['mismatches']
    assert evidence['online']['fixed_synthetic_paths']==evidence['online']['reverse_order_paths_exact']==90
    assert evidence['training']['inner_reference_models']==30 and evidence['training']['fitted_heads']==25
    frozen=read_json(BASE/'FINAL_CANDIDATE_FREEZE_NEXT7.json')
    assert frozen['status']=='FROZEN' and frozen['decision']=='NEXT7_NO_PROMOTION' and not frozen['selected_NEXT7_roles']
    for mapping in ('research_bundle_files_sha256','NEXT7_inference_and_library_sources_sha256','preserved_historical_reference_files_sha256'):
        for name,sha in frozen[mapping].items():assert digest(ROOT/name)==sha,name
    warm=rows(BASE/'LATENCY_NEXT7.csv')
    direct=rows(BASE/'DIRECT_10M_THROUGHPUT_NEXT7.csv')
    assert len(direct)==14 and all(r['points']=='10000000' and r['numeric_state_size_stable']=='True' for r in direct)
    assert evidence['runtime']['direct_calls']==140000000
    comparisons=rows(BASE/'MODEL_COMPARISON_NEXT7.csv')
    candidates=[r for r in comparisons if r['new_NEXT7_scientific_candidate']=='True']
    assert len(candidates)==15 and all(float(r['mean_matched_delta_M0'])<0 and float(r['worst_matched_fold_delta_M0'])<0 for r in candidates)
    best=max(candidates,key=lambda r:float(r['mean_TS_AUC']))
    with activity('Consolidate final data, lineage, runtime and reproduction audit','FINAL_AUDIT','FINAL_AUDIT'):
        byte_scope={'status':'PASS','created_utc':utc_now(),'measurements':warm,
            'direct_long_and_repeated_10M_workloads':direct,
            'numeric_array_bytes_unchanged_in_direct_workloads':True,
            'memory_scope':'Reported numeric arrays exclude immutable fitted trees and Python object overhead. For NEXT7 classifier additions the bounded_mutable field describes the added state branch; it is not a whole-process RSS bound.',
            'current_and_event_state_distinct':True,'G2_new_qualification_after16384':'ABSTAIN_WITH_PREVIOUS_MEMORY_RETAINED',
            'strict_all_raw_input_disjointness_in_inherited_core':'NOT_MET_LEGACY_LAG4_FOOTPRINT',
            'latency_done_sha256':digest(BASE/paths['runtime'])}
        save('STATE_MEMORY_NEXT7.json',byte_scope)
        online=evidence['online']
        save('PREFIX_INVARIANCE_NEXT7.json',{'status':'PASS','created_utc':utc_now(),
            'online_contract_receipt_sha256':digest(BASE/paths['online']),
            'paths_checked':90,'prefix_lengths':[1,16,64,257],
            'changed_future_suffix_and_total_length_prefix_bitwise_equal':True,
            'forbidden_tau_future_length_future_suffix_dataset_id_arguments_rejected':True,
            'scope':'Declared callable scalar interfaces and original fixed synthetic audit paths, with full regression evidence. No claim to an OS-enforced information-flow sandbox.'})
        save('DETERMINISM_NEXT7.json',{'status':'PASS','created_utc':utc_now(),
            'online_receipt_sha256':digest(BASE/paths['online']),'fixed_fixture_scalar_paths_bitwise_equal':90,
            'reverse_order_reset_paths_bitwise_equal':90,'save_load_and_interleaved_state_continuations_bitwise_equal':True,
            'offline_batch_scalar_tolerance':1e-12,'offline_batch_may_differ_in_last_float_bits':True,
            'all_per_path_max_errors_recorded_in':'audit/online_contracts/',
            'environment_scope':'Current recorded Windows/Python/library environment; cross-hardware bitwise identity is not promised.'})
        save('LEAKAGE_AUDIT_NEXT7.json',{'status':'PASS','created_utc':utc_now(),
            'data_receipt_sha256':digest(BASE/paths['data']),'nested_training_receipt_sha256':digest(BASE/paths['training']),
            'online_contract_receipt_sha256':digest(BASE/paths['online']),
            'research_membership_sha256':evidence['data']['research_split_sha256'],
            'RP_only_series':6400,'original_grouped_folds':5,'strict_inner_reference_fits':30,
            'outer_validation_used_in_correction_fit':False,'tau_or_future_or_dataset_id_in_inference':False,
            'SHADOW_observation_rows_decoded':0,'SEAL_observation_rows_decoded':0,
            'reduced_public_cloud_submission_used':False,
            'truth_use':'RP scoring and explicitly offline mechanistic/time-age diagnosis; synthetic truth labels only annotate completed paths.',
            'guard_scope':'Application-level admission plus recorded read audits, not an OS sandbox.'})
        save('REPRODUCTION_NEXT7.json',{'status':'PASS','created_utc':utc_now(),
            'selected_new_NEXT7_roles':[],'fresh_raw_RP_training_refit':'NOT_APPLICABLE_NO_NEW_ROLE',
            'no_meaningless_refit_of_old_primary':True,
            'isolated_source_model_payload_reproduction':evidence['reproduction'],
            'historical_reference_public_entry_reproduction':evidence['historical_reference_reproduction'],
            'final_online_contract_receipt_sha256':digest(BASE/paths['online']),
            'source_and_model_fingerprints_rechecked_during_final_audit':True,
            'archive_extraction_reproduction':'Required delivery postlude, bound to exact final ZIP by external FINAL_DELIVERY_AUDIT_NEXT7.json; not claimed complete by this pre-archive receipt.'})
        decisions=[]
        for row in candidates:
            decisions.append({'method':row['method'],'final_decision':'NO_PROMOTION',
                'mean_matched_delta_M0':row['mean_matched_delta_M0'],'worst_matched_fold_delta_M0':row['worst_matched_fold_delta_M0'],
                'performance_and_generation_mean_gate':'FAIL','robustness_worst_fold_gate':'FAIL',
                'runtime_measurement_available':any(r['method']==row['method'] for r in warm),
                'runtime_cannot_override_predictive_gate':True,'ALT':'NOT_RUN_NO_STRONG_CANDIDATE',
                'same_frozen_gates_used':True})
        csv_write('FINAL_CANDIDATE_DECISIONS_NEXT7.csv',decisions)
        environment={'status':'RECORDED','created_utc':utc_now(),'python':sys.version,'platform':platform.platform(),
            'packages':{name:metadata.version(name) for name in ('numpy','numba','llvmlite','scipy','scikit-learn',
                'lightgbm','xgboost','joblib','threadpoolctl','pandas','pyarrow','psutil','pytest')},
            'inference_math_threads':1,'figure_environment':'Separate .figure-venv; matplotlib plots are data-derived.',
            'hardware_independent_runtime_or_bitwise_guarantee':False}
        save('ENVIRONMENT_NEXT7.json',environment)
        source_paths=[p for p in (ROOT/'src').rglob('*.py') if '__pycache__' not in p.parts]
        source_paths+=list((ROOT/'scripts/next7').glob('*.py'))
        source_paths+=list((ROOT/'scripts/next7').glob('*.mjs'))
        source_paths+=list((ROOT/'src/next7').glob('*.html'))
        source_paths+=[ROOT/name for name in ('requirements.txt','pyproject.toml','COUNTEREXAMPLE_SUITE_LOCK.json')]
        source_paths+=[ROOT/'configs'/name for name in ('features.json','arch_input.json','ar_residual_input.json',
            'historical_context.json','historical_quantiles.json','magnitude_dependence.json','rank_gaussian.json')]
        files={p.relative_to(ROOT).as_posix():digest(p) for p in sorted(set(source_paths))}
        files.update(frozen['research_bundle_files_sha256'])
        files.update(frozen['preserved_historical_reference_files_sha256'])
        role=frozen['selected_final_primary'];files[role['model_file']]=digest(ROOT/role['model_file'])
        if (ROOT/role['entry']).is_file():files[role['entry']]=digest(ROOT/role['entry'])
        save('SOURCE_MANIFEST_NEXT7.json',{'status':'FROZEN','created_utc':utc_now(),'files_sha256':files,
            'scope':'Source, required configurations and retained research bundle/model bytes. Result documents and this manifest are excluded to avoid hash cycles. The ZIP has its own complete member-byte manifest.',
            'original_protected_authority_manifest_sha256':digest(BASE/'PROTECTED_BEFORE_NEXT7_SHA256.json'),
            'environment_sha256':digest(BASE/'ENVIRONMENT_NEXT7.json')})
        board=read_json(BASE/'runtime/BOTTLENECK_BOARD_CURRENT.json')
        save('BOTTLENECK_BOARD_FINAL_NEXT7.json',{**board,'status':'RESEARCH_CLOSED_NO_PROMOTION',
            'finalized_utc':utc_now(),'top_three':['H-reference transport and long-null lifetime-memory validity',
                'False versus true event rank tradeoff under target persistence','Gains concentrated in limited mechanisms/time regions rather than overall matched ranking'],
            'next_three':['Design a consistent H-reference calibration with an explicit serial-dependence/horizon validity target',
                'Preregister a small correction that separates event admission error from irreversible target memory',
                'Evaluate an independently justified scale/contraction signal with the same nested groups and fixed decision gates'],
            'no_more_threshold_or_recipe_search_in_this_session':True})
        ledger=rows(BASE/'EXPERIMENT_LEDGER_NEXT7.csv')
        output_names=['STATE_MEMORY_NEXT7.json','PREFIX_INVARIANCE_NEXT7.json','DETERMINISM_NEXT7.json',
            'LEAKAGE_AUDIT_NEXT7.json','REPRODUCTION_NEXT7.json','FINAL_CANDIDATE_DECISIONS_NEXT7.csv',
            'ENVIRONMENT_NEXT7.json','SOURCE_MANIFEST_NEXT7.json','BOTTLENECK_BOARD_FINAL_NEXT7.json']
        save('FINAL_AUDIT_NEXT7.json',{'status':'PASS','created_utc':utc_now(),'decision':'NEXT7_NO_PROMOTION',
            'scope':'Completed research and computation audit. Exact final ZIP extraction/replay and desktop-copy completion are bound by external FINAL_DELIVERY_AUDIT_NEXT7.json.',
            'productive_research_seconds':productive,'final_audit_seconds':audit,
            'final_audit_target_seconds':3600,'final_audit_target_is_approximate':True,
            'wall_seconds_at_computation_audit':time.time()-session['started_epoch'],
            'idle_time_credited':False,'parallel_work_double_counted':False,
            'scientific_full_candidates':15,'screen_candidates':7,'major_families':6,'learned_recipes':5,
            'new_inner_reference_fits':30,'new_correction_head_fits':25,'new_model_fits_total':55,
            'ledger_work_units_at_snapshot':len(ledger),'ledger_units_are_not_independent_scientific_candidates':True,
            'start_primary':'FROZEN_BL80','final_primary':'FROZEN_BL80','promoted_primary_delta':0.,
            'best_NEXT7_research_candidate':best,'new_generation':False,'new_robustness_model':False,
            'selected_NEXT7_roles':[],'ALT_opened':False,'SHADOW_opened':False,'SEAL_opened':False,
            'known_unmet_design_requirements':['Inherited core lag4 features do not meet strict all-raw-input block nonoverlap. New signed-scale raw blocks are disjoint; event-update mechanics and finite influence do not waive the core requirement.'],
            'reduced_public_cloud_used':False,'unchanged_protected_files':6333,
            'complete_regression':evidence['regression'],'exact_nested_lineage_audited':True,
            'raw_data_and_group_membership_verified':True,'direct_scalar_latency_calls':140000000,
            'audit_evidence_sha256':{name:digest(BASE/name) for name in paths.values()},
            'audit_outputs_sha256':{name:digest(BASE/name) for name in output_names},
            'candidate_freeze_sha256':digest(BASE/'FINAL_CANDIDATE_FREEZE_NEXT7.json'),
            'remaining_delivery_work':['Write final45-question Korean guide and GPT handoff','Freeze completed monitor ledger',
                'Build final ZIP, verify its fresh extracted replay, copy exactly20 desktop files'],
            'no_predictive_promotion_from_synthetic_only_or_runtime_results':True})
    note('Final computation audit consolidated','No promotion; all required scientific and engineering checks complete','FINAL_AUDIT',phase='FINAL_AUDIT')


if __name__=='__main__':main()
