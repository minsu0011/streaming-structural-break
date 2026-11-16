"""Freeze the negative candidate decision only after completed productive research."""
from pathlib import Path
import argparse
import csv
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.next7.common import BASE,read_json,save,digest,utc_now,union_seconds,activity,note,require_guard
from src.next6_extension.synthetic import CASES

RESEARCH_FILES = '''NEXT7_PROTOCOL_LOCK.json NEXT7_CANDIDATE_BUDGET.json NEXT7_RANDOMNESS_LOCK.json
CURRENT_STATE_CONTRACT.md CONFIRMATION_ENGINE_CONTRACT.md EVENT_MEMORY_CONTRACT.md
CONTRACT_AMENDMENT_G2_NEXT7_KO.md SIGNED_SCALE_FEATURES.csv H_CALIBRATION_RESULTS.csv
CONFIRMATION_RULE_RESULTS.csv EVENT_MEMORY_RESULTS.csv MODEL_COMPARISON_NEXT7.csv CV_RESULTS_NEXT7.csv
PAIRED_BOOTSTRAP_NEXT7.csv ALT_RESULTS_NEXT7.csv REGIME_RESULTS_NEXT7.csv TIME_FRACTURE_NEXT7.csv
BREAK_AGE_NEXT7.csv SCALE_EXPANSION_CONTRACTION_RESULTS.csv SHOCK_RESULTS_NEXT7.csv
LONG_NULL_RESULTS_NEXT7.csv FALSE_LATCH_RESULTS_NEXT7.csv EVENT_TIMING_FORENSIC.csv
TARGET_ALIGNMENT_RESULTS.csv CLASSIFIER_LONG_NULL_RESULTS_NEXT7.csv CLASSIFIER_AMPLITUDE_CONTROL_NEXT7.csv
CONDITIONAL_NULL_RESULTS_NEXT7.csv EMPIRICAL_SUPPORT_PARTITION_NEXT7.csv CROSSED_H_LENGTH_CONTRASTS_NEXT7.csv
H_NORMALIZER_RISK_ASSOCIATIONS_NEXT7.csv G3_ALL_PAIRS_OBJECTIVE_DIAGNOSTICS_NEXT7.csv
SCORE_EVIDENCE_CONSOLIDATED.json MEDIAN_ESTIMAND_CLARIFICATION_NEXT7.json
COMPLETED_CORE_TABLES_NEXT7.json CONFIRMATION_RULE_REPORT_DONE.json CONFIRMATION_RULE_EQUIVALENCE_DONE.json
RANK_REFERENCE_GEOMETRY_AUDIT_DONE.json SHOCK_INFLUENCE_AUDIT_DONE.json COMPLETE_DIAGNOSTICS_GRADIENTS_NEXT7.json
SCALE_ADMISSION_CEILING_DONE_NEXT7.json OBSERVED_SCALE_ADMISSION_DONE_NEXT7.json SCALE_EVIDENCE_PATH_AUDIT_DONE_NEXT7.json
visuals/scale_evidence/original_cases_001/DONE.json audit/SCALE_EVIDENCE_VISUAL_REVIEW_001.json
visuals/classifier_scale/original_cases_002/DONE.json audit/CLASSIFIER_SCALE_VISUAL_REVIEW_002.json
visuals/classifier_stress/full_suite_001/MANIFEST.json audit/CLASSIFIER_FULL_SUITE_VISUAL_REVIEW_001.json
CROSSED_MOMENT_AUDIT_DONE_NEXT7.json CROSSED_RESAMPLING_EXACT_MOMENTS_NEXT7.csv
ROLE_GATE_FEASIBILITY_DONE_NEXT7.json ROLE_GATE_EMPIRICAL_FEASIBILITY_NEXT7.csv
FRESH_BLOCK_REQUIREMENT_AUDIT_NEXT7.json
audit/O1_CLASSIFIER_MEMORY_BOUNDARY_REVIEW_001.json research_notes/CLASSIFIER_SHOCK_WITHOUT_EVENT_MEMORY_KO.md
SHOCK_CLASSIFIER_JOINT_AUDIT_LOCK_NEXT7.json SHOCK_CLASSIFIER_JOINT_AUDIT_DONE_NEXT7.json
SHOCK_CLASSIFIER_JOINT_PROBABILITIES_NEXT7.csv SHOCK_CLASSIFIER_ENDPOINT_DECOMPOSITION_NEXT7.csv
research_notes/PAIRED_CONTROL_ALARM_ENDPOINT_INTERPRETATION_KO.md
visuals/diagnostics/final_all_001/MANIFEST.json audit/FINAL_DIAGNOSTIC_VISUAL_REVIEW_001.json
visuals/normalizer_transport/final_all_001/MANIFEST.json audit/FINAL_NORMALIZER_VISUAL_REVIEW_001.json
COMPLETE_DIAGNOSTICS_CLASSIFIERS_NEXT7.json COMPLETE_DIAGNOSTICS_CONDITIONAL_NEXT7.json
G1_ALL_OUTER_PREDICTIONS_FROZEN.json G2_ALL_OUTER_PREDICTIONS_FROZEN.json G3_ALL_OUTER_PREDICTIONS_FROZEN.json
G1_OUTER_SCORE_RELEASE.json G2_OUTER_SCORE_RELEASE.json G3_OUTER_SCORE_RELEASE.json
G1_BOOTSTRAP_DONE.json G2_BOOTSTRAP_DONE.json G3_BOOTSTRAP_DONE.json
FRACTURE_G1_G2_DONE.json FRACTURE_G3_DONE.json SYNTHETIC_UNCERTAINTY_DONE_NEXT7.json
RESEARCH_BUNDLES_EXPORTED.json reproduction/research_002/DONE.json
FROZEN_REFERENCE_DELIVERY_LOCK_NEXT7.json frozen_reference_fixture/FIXTURE.json audit/FROZEN_REFERENCE_ISOLATED_REPLAY_001.json
visuals/v3/BROWSER_REVIEW.json audit/VIEWER_V3_VISUAL_REVIEW_001.json
visuals/v4/BUILD_DONE.json visuals/v4/BROWSER_REVIEW.json audit/VIEWER_V4_VISUAL_REVIEW_001.json
audit/OWNED_CRASH_RECOVERY_TESTS_001.json'''.split()


def csv_rows(path):
    with path.open(newline='',encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def pending():
    queue=read_json(BASE/'runtime/TASK_QUEUE.json')
    jobs=[item['id'] for item in queue if not (BASE/'runtime/queue'/item['id']/'TERMINAL.json').exists()]
    files=[name for name in RESEARCH_FILES if not (BASE/name).is_file()]
    return queue,jobs,files


def main(inspect_only):
    queue,jobs,missing=pending()
    heartbeat=read_json(BASE/'runtime/HEARTBEAT.json')
    if inspect_only:
        print({'productive_seconds':heartbeat['productive_seconds'],'research_minimum_met':heartbeat['research_minimum_met'],
            'queue_tasks':len(queue),'outstanding_queue_tasks':len(jobs),'missing_required_research_files':missing})
        return
    require_guard()
    state=read_json(BASE/'runtime/SCHEDULER_STATE.json')
    session=read_json(BASE/'SESSION_NEXT7.json')
    assert time.time()-state['last_epoch']<90
    assert union_seconds(state['intervals'])>=session['research_minimum_seconds'], 'Productive minimum not yet met'
    assert not state['audit_intervals'] and not (BASE/'runtime/FINAL_AUDIT_ADMISSION.json').exists()
    assert not jobs and not missing, 'All scientific and diagnostic tasks must be completed first'
    for item in queue:
        path=BASE/'runtime/queue'/item['id']/'TERMINAL.json'
        terminal=read_json(path)
        if terminal['status']!='COMPLETE':
            resolved=read_json(path.parent/'RESOLUTION.json')
            assert resolved['status'] in ('REPAIRED_AND_REPRODUCED','EXPECTED_FAILURE_TEST')
            assert resolved['terminal_sha256']==digest(path)
            for name,sha in resolved['evidence_sha256'].items():assert digest(BASE/name)==sha
    for kind in ('GRADIENTS','CLASSIFIERS','CONDITIONAL'):
        receipt=read_json(BASE/f'COMPLETE_DIAGNOSTICS_{kind}_NEXT7.json')
        assert receipt['status']=='COMPLETE' and not receipt['partial_cohorts_included']
        for name,sha in receipt['outputs_sha256'].items():assert digest(BASE/name)==sha
    for case in CASES:
        assert read_json(BASE/f'classifier_statistics/{case}/DONE.json')['status']=='COMPLETE'
    for case in ('N0','N1','N2','N3'):
        for length in (1000,2048,5000):
            for group in ('conditional_null/grid','conditional_statistics','support_partition','normalizer_transport/correlations'):
                assert read_json(BASE/f'{group}/{case}/H_{length}/DONE.json')['status']=='COMPLETE'
        assert read_json(BASE/f'crossed_H_length_comparison/{case}/DONE.json')['status']=='COMPLETE'
    moments=read_json(BASE/'CROSSED_MOMENT_AUDIT_DONE_NEXT7.json')
    assert moments['status']=='COMPLETE' and moments['cohorts']==12 and moments['rows']==108
    assert not moments['partial_cohorts_included'] and not moments['original_intervals_replaced']
    assert digest(BASE/'CROSSED_RESAMPLING_EXACT_MOMENTS_NEXT7.csv')==moments['output_sha256']
    feasibility=read_json(BASE/'ROLE_GATE_FEASIBILITY_DONE_NEXT7.json')
    assert feasibility['status']=='COMPLETE' and feasibility['all15_candidates_fail_user_required_worst_fold_nonworse_independently']
    assert not feasibility['gate_changed']
    assert digest(BASE/'ROLE_GATE_EMPIRICAL_FEASIBILITY_NEXT7.csv')==feasibility['output_sha256']
    block_contract=read_json(BASE/'FRESH_BLOCK_REQUIREMENT_AUDIT_NEXT7.json')
    assert block_contract['status']=='COMPLETE'
    assert block_contract['inherited_core_all_raw_input_footprints']=='NOT_MET_PREVIOUS_RAW_LAG_FOOTPRINT'
    assert not block_contract['strict_nonoverlap_production_readiness_claim']
    shock_boundary=read_json(BASE/'audit/O1_CLASSIFIER_MEMORY_BOUNDARY_REVIEW_001.json')
    assert shock_boundary['status']=='PASS' and shock_boundary['exact_alarm_arrays_match_reviewed_viewer']
    assert not shock_boundary['new_fit_or_seed_or_draw_or_threshold_or_candidate']
    assert shock_boundary['source_sha256']==digest(ROOT/'scripts/next7/review_classifier_shock_boundary.py')
    for name,sha in shock_boundary['input_sha256'].items():
        assert digest(BASE/name)==sha
    shock_joint=read_json(BASE/'SHOCK_CLASSIFIER_JOINT_AUDIT_DONE_NEXT7.json')
    assert shock_joint['status']=='COMPLETE' and shock_joint['complete_cases']==3
    assert shock_joint['joint_rows']==48 and shock_joint['comparison_rows']==42
    assert shock_joint['missing_joint_rows_preserved']==6
    assert not shock_joint['new_draws_or_intervals_or_models_or_thresholds']
    for mapping in ('inputs_sha256','outputs_sha256'):
        for name,sha in shock_joint[mapping].items():
            assert digest(BASE/name)==sha
    assert read_json(BASE/'SHOCK_CLASSIFIER_JOINT_AUDIT_LOCK_NEXT7.json')['source_sha256']==digest(ROOT/'scripts/next7/review_shock_classifier_joint.py')
    for name in ('audit/CLASSIFIER_SCALE_VISUAL_REVIEW_002.json',
                 'audit/CLASSIFIER_FULL_SUITE_VISUAL_REVIEW_001.json',
                 'audit/FINAL_DIAGNOSTIC_VISUAL_REVIEW_001.json',
                 'audit/FINAL_NORMALIZER_VISUAL_REVIEW_001.json'):
        assert read_json(BASE/name)['status']=='PASS'
    normalizer_figure=read_json(BASE/'visuals/normalizer_transport/final_all_001/MANIFEST.json')
    assert len(normalizer_figure['completed_cohorts'])==12
    classifier_figure=read_json(BASE/'visuals/classifier_stress/full_suite_001/MANIFEST.json')
    assert classifier_figure['status']=='COMPLETE' and set(classifier_figure['completed_cases'])==set(CASES)
    assert classifier_figure['displayed_cells']==368 and classifier_figure['missing_cells']==8
    candidates=[r for r in csv_rows(BASE/'MODEL_COMPARISON_NEXT7.csv') if r['new_NEXT7_scientific_candidate']=='True']
    assert len(candidates)==15
    assert all(float(r['mean_matched_delta_M0'])<0 for r in candidates)
    assert all(float(r['worst_matched_fold_delta_M0'])<0 for r in candidates)
    usage=read_json(BASE/'runtime/CANDIDATE_USAGE.json')
    assert usage['full_completed']==usage['full_reserved']==15
    assert usage['learned_meta_reserved']==5 and usage['major_families_reserved']==6
    alt=csv_rows(BASE/'ALT_RESULTS_NEXT7.csv')
    assert len(alt)==15 and all(r.get('status')=='NOT_RUN' for r in alt)
    role=read_json(BASE/'RESTORED_NEXT6_AUTHORITY.json')['frozen_roles']['FROZEN_PRIMARY']
    assert digest(ROOT/role['model_file'])==role['model_sha256']
    references=read_json(BASE/'FROZEN_REFERENCE_DELIVERY_LOCK_NEXT7.json')
    for name,sha in references['files_sha256'].items():assert digest(ROOT/name)==sha
    reference_replay=read_json(BASE/'audit/FROZEN_REFERENCE_ISOLATED_REPLAY_001.json')
    assert reference_replay['status']=='PASS' and reference_replay['public_entry_paths']==9
    with activity('Final research candidate freeze and completion inventory','RESEARCH_CLOSURE','REPRODUCTION_AND_RUNTIME'):
        model_files=sorted((BASE/'research_bundles').glob('fold_*/*'))
        model_hashes={p.relative_to(ROOT).as_posix():digest(p) for p in model_files if p.is_file()}
        source_files=sorted((ROOT/'src/next7').glob('*.py'))
        frozen={'status':'FROZEN','created_utc':utc_now(),'decision':'NEXT7_NO_PROMOTION',
            'productive_seconds_at_freeze':union_seconds(state['intervals']),
            'productive_minimum_seconds':session['research_minimum_seconds'],
            'scientific_full_candidates':15,'learned_recipes':5,'major_families':6,'screen_candidates':7,
            'scientific_success_scope':['E_FAILURE_LOCATION_NARROWED','F_STRONG_NEGATIVE_RESULT'],
            'selected_NEXT7_roles':[], 'selected_final_primary':role,
            'original_robustness_and_distilled_references_retained':True,
            'score_table_sha256':digest(BASE/'MODEL_COMPARISON_NEXT7.csv'),
            'candidate_gate_failures':[{'method':r['method'],'mean_delta':float(r['mean_matched_delta_M0']),
                'worst_matched_fold_delta':float(r['worst_matched_fold_delta_M0']),
                'performance_and_generation_mean_gate':'FAIL','robustness_worst_fold_gate':'FAIL'} for r in candidates],
            'ALT':'NOT_RUN_NO_STRONG_CANDIDATE','new_raw_RP_training_reproduction':'NOT_APPLICABLE_NO_NEW_ROLE',
            'diagnostic_bundle_scope':'Three negative recipes I4/G2_FULL/G3 across five already fitted research folds, plus their common base models. These are reproducibility evidence, not selected production roles or15 retained candidates.',
            'research_bundle_files_sha256':model_hashes,
            'preserved_historical_reference_files_sha256':references['files_sha256'],
            'historical_reference_delivery_lock_sha256':digest(BASE/'FROZEN_REFERENCE_DELIVERY_LOCK_NEXT7.json'),
            'NEXT7_inference_and_library_sources_sha256':{p.relative_to(ROOT).as_posix():digest(p) for p in source_files},
            'remaining_work':'Final audit, measured runtime, full tests, guarded reproduction and delivery. No new model search.'}
        save('FINAL_CANDIDATE_FREEZE_NEXT7.json',frozen)
        save('RESEARCH_COMPLETION_READY.json',{'status':'PASS','created_utc':utc_now(),
            'source_sha256':digest(__file__),'outstanding_scientific_tasks':0,
            'candidate_freeze_sha256':digest(BASE/'FINAL_CANDIDATE_FREEZE_NEXT7.json'),
            'required_artifacts_sha256':{name:digest(BASE/name) for name in RESEARCH_FILES},
            'queue_terminal_sha256':{item['id']:digest(BASE/'runtime/queue'/item['id']/'TERMINAL.json') for item in queue},
            'fixed_null_cohorts_completed':12,'classifier_cases_completed':len(CASES),
            'all_comparisons_complete_before_final_audit':True})
    note('Research closure ready for final audit','No promotion; all registered science and diagnostics complete','REPRODUCTION_AND_RUNTIME')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--inspect',action='store_true')
    main(parser.parse_args().inspect)
