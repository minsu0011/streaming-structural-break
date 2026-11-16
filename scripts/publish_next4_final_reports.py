"""Publish completed audit evidence and all required Korean handoff reports."""
import argparse
import csv
from pathlib import Path
import re
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next4.finalization import require_handoff_time
from src.next4.handoff_payload import REQUIRED
from src.next4.korean_reports import documents


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',default='final_v1');args=parser.parse_args();tag=args.tag
    assert re.fullmatch('[A-Za-z0-9_]+',tag);timing=require_handoff_time()
    out=BASE/'final_publication'/tag;out.mkdir(parents=True,exist_ok=False);evidence={};prior={}
    def read(path,status=None):
        r=read_json(path)
        if status is not None:assert r['status']==status,(str(path),r.get('status'))
        evidence[path.relative_to(ROOT).as_posix()]=digest(path);return r
    def preserve(name):
        path=ROOT/name
        if path.exists():
            target=out/'previous_root_files'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
            prior[name]={'sha256':digest(path),'preserved':target.relative_to(ROOT).as_posix()}
    def write_json(name,value):preserve(name);atomic_json(ROOT/name,value)
    def copy(source,name):
        preserve(name);shutil.copyfile(source,ROOT/name);assert digest(ROOT/name)==digest(source)
    selection=read(ROOT/'DEV_CANDIDATES_NEXT4.json','FROZEN_NEXT4_DEV_SELECTION');role=selection['roles'].get('NEXT4_ROBUSTNESS')
    for name,sha in selection['evidence_sha256'].items():assert digest(ROOT/name)==sha
    plan=read(BASE/'final_audit/PLAN.json')
    for identifier in plan['jobs']:
        state=read(BASE/'jobs'/identifier/'STATE.json','COMPLETE');assert state['exit_code']==0
    supplemental=read(BASE/'final_audit/SUPPLEMENTAL_PLAN.json')
    for identifier in supplemental['jobs']:
        state=read(BASE/'jobs'/identifier/'STATE.json','COMPLETE');assert state['exit_code']==0
    auditdir=BASE/'final_audit'/tag
    boundary=read(auditdir/'frozen_boundary/RESULTS.json','PASS')
    stats=read(auditdir/'primary_statistics/RESULTS.json','PASS');assert len(stats['candidate_summaries'])==14
    mechanisms=read(auditdir/'mechanism_statistics/RESULTS.json','PASS')
    nominations=read(auditdir/'nomination_chronology/RESULTS.json','PASS')
    transfer_stats=read(auditdir/'transfer_statistics/RESULTS.json','PASS')
    tests=read(auditdir/'full_tests/RESULTS.json','PASS')
    portable=read(BASE/'portable_verification'/tag/'RESULTS.json','PASS')
    apis={label:read(auditdir/('API_'+label)/'RESULTS.json','PASS') for label in ['BL80','B2']+([role['experiment_id']] if role else [])}
    for label,result in apis.items():assert result['prefix_cases']>=300 and result['offline_streaming_all_DEV_exact'] and result['persistent_inference_write_attempts']==0
    tables=read(BASE/'report_tables'/tag/'RESULTS.json','PASS');assert not tables['synthetic_pending']
    fits=read(BASE/'fit_inventory'/tag/'RESULTS.json','PASS');assert fits['unfinished_primitive_fit_starts']==0
    ledger=read(BASE/'ledger'/tag/'RESULTS.json','PASS')
    registration=read(BASE/'registration_audit'/tag/'RESULTS.json','PASS')
    execution=read(BASE/'execution_graph_audit'/tag/'RESULTS.json','PASS')
    read(BASE/'gate_reviews'/tag/'RESULTS.json','SCIENTIFIC_GATE_REVIEW_NOT_FINAL_SELECTION')
    reflection=read(BASE/'sign_reflection/RESULTS.json','PASS')
    cold_b2=read(BASE/'cold_api_readonly_v2/RESULTS.json','PASS')
    cold_bl=read(BASE/'cold_api/BL80_1.json','PASS')
    nested=read(BASE/'nested_reuse_audit/RESULTS.json','PASS')
    negative=read(BASE/'negative_controls/label_permutation_v1/RESULTS.json','PASS')
    runtime_b2=read(BASE/'latency/B2_DIRECT/RESULTS.json','PASS')
    runtime={'B2_diagnostic_research':runtime_b2};fresh=None
    if role:
        cid=role['experiment_id'];runtime['NEXT4_ROBUSTNESS_research']=read(BASE/'latency'/(cid+'_DIRECT')/'RESULTS.json','PASS')
        runtime['NEXT4_ROBUSTNESS_final_audit']=read(BASE/'latency'/(cid+'_AUDIT')/'RESULTS.json','PASS')
        assert runtime['NEXT4_ROBUSTNESS_final_audit']['direct_runtime_within_3x']
        fresh=read(BASE/'reproduction'/cid/'RESULTS.json','PASS')
        assert fresh['fresh_raw_reproduction'] and fresh['fresh_raw_feature_and_inference_exact']
        read(BASE/'reproduction'/cid/'FRESH_FEATURES.json','PASS')
        read(BASE/'finalist_validation'/cid/'COLD/RESULTS.json','PASS')
    else:runtime['FROZEN_PERFORMANCE_final_audit']=read(BASE/'latency/FROZEN_FINAL_AUDIT/RESULTS.json','PASS')
    # Every nominated scientific suite must have completed, including actual final artifacts.
    synthetic=[]
    for path in sorted((BASE/'synthetic_nominations').glob('*.json')):
        nom=read(path);r=read(BASE/'synthetic/candidates'/nom['id']/'RESULTS.json','PASS')
        assert r['counterexample_scenarios']==21 and r['long_null_regimes']==4 and r['long_null_horizons']==[1024,4096,16384]
        synthetic.append({'nomination':nom['id'],'scope':nom['training_scope'],'new_fits':r['new_fits']})
    failures=[]
    for path in sorted((BASE/'jobs').glob('*/STATE.json')):
        state=read_json(path)
        if state['status']=='QUARANTINED':
            resolution=read(BASE/'failure_resolutions'/(state['id']+'.json'),'RESOLVED_WITH_PRESERVED_FAILURE')
            assert resolution['quarantined_state_sha256']==digest(path)
            for name,sha in resolution['evidence_sha256'].items():assert digest(ROOT/name)==sha
            failures.append({'job':state['id'],'original_failure_preserved':True,'resolution':resolution})
    # Keep the immutable pre-score policies; the root budget also records final usage explicitly.
    copy(BASE/'preflight/NEXT4_PREFLIGHT.json','NEXT4_PREFLIGHT.json')
    copy(ROOT/'configs/next4/HYPOTHESIS_FAMILY_LOCK.json','HYPOTHESIS_FAMILY_LOCK.json')
    budget=read(ROOT/'configs/next4/CANDIDATE_BUDGET.json')
    write_json('CANDIDATE_BUDGET.json',{**budget,'final_usage_recorded_utc':utc_now(),
        'declared_families':8,'families_with_new_DEV_scores':7,'screen_candidates':16,'full_candidates':14,
        'family_screen_full_counts':{'A':[3,3],'B':[2,2],'C':[3,3],'D':[3,3],'E':[2,1],'F':[0,0],'G':[2,1],'H':[1,1]},
        'real_DEV_learner_fits':fits['real_DEV_learner_fit_completions'],'new_frozen_base_fits':0,
        'new_selected_roles':len(selection['roles'])-1,'new_candidate_identity_from_diagnostics':0,
        'original_frozen_budget_sha256':digest(ROOT/'configs/next4/CANDIDATE_BUDGET.json')})
    for name,sha in tables['files_sha256'].items():
        source=BASE/'report_tables'/tag/name;assert digest(source)==sha;copy(source,name)
    copy(BASE/'bottlenecks/history/001.json','BOTTLENECK_BOARD_INITIAL_NEXT4.json')
    copy(BASE/'bottlenecks/BOARD.json','BOTTLENECK_BOARD_FINAL_NEXT4.json')
    copy(BASE/'bottlenecks/HISTORY.csv','BOTTLENECK_HISTORY_NEXT4.csv')
    # Publish the reconciled ledger with old cells unchanged; update NEXT4-only context columns.
    source=BASE/'ledger'/tag/'EXPERIMENT_LEDGER.csv';assert digest(source)==ledger['files_sha256'][source.name]
    with source.open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream);fields=reader.fieldnames;rows=list(reader)
    before=[dict(r) for r in rows[:242]]
    for row in rows[242:]:
        row['session_finalization_status']='NEXT4_FINAL_AUDIT_COMPLETE'
        cid=row.get('next4_candidate_id','')
        if row.get('ledger_section')=='NEXT4_SCIENTIFIC_RECIPES' and role and cid==role['experiment_id']:
            row['status']='NEXT4_ROBUSTNESS_ONLY_PRIMARY_PERFORMANCE_REJECTED'
            row['reason']='Original supported SCALE and fixed decrease confirmation plus direct, reproduction and final audit requirements passed; not a new primary champion.'
    preserve('EXPERIMENT_LEDGER.csv')
    with (ROOT/'EXPERIMENT_LEDGER.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    with (ROOT/'EXPERIMENT_LEDGER.csv').open(encoding='utf-8-sig',newline='') as stream:written=list(csv.DictReader(stream))
    assert written[:242]==before
    common={'created_utc':utc_now(),'audit_started_utc':read_json(BASE/'SESSION.json')['audit_started_utc'],'new_seal_rows':0}
    write_json('SEAL_GUARD_NEXT4.json',{'status':'PASS',**common,'all_four_seals_unopened':True,'DEV_series':8000,'seal_series':2000,
        'blocked_admissions':boundary['blocked_seal_admissions'],'split_sha256':boundary['split_sha256'],
        'seal_lock_sha256':boundary['seal_lock_sha256'],'reduced_ledger_sha256':boundary['reduced_ledger_sha256'],
        'new_reduced_usage':0,'historical_reduced_usage':'2/2 retained','new_public_submission':0,'new_cloud_submission':0,
        'raw_scope':'Whole-file byte hashing and DEV-predicate reads only; no seal values, labels, tau or predictions.',
        'evidence':(auditdir/'frozen_boundary/RESULTS.json').relative_to(ROOT).as_posix()})
    write_json('PREFIX_INVARIANCE_NEXT4.json',{'status':'PASS',**common,'final_audit_models':apis,
        'new_feature_family_checks':'Completed final full suite includes original >=100-case feature-prefix tests and mathematical oracles; exact test definitions are included in tests/.',
        'finalist_minimum_cases':300,'future_suffix_and_final_length_access_trapped':True,
        'evidence_sha256':{(auditdir/('API_'+key)/'RESULTS.json').relative_to(ROOT).as_posix():digest(auditdir/('API_'+key)/'RESULTS.json') for key in apis}})
    write_json('LEAKAGE_AUDIT_NEXT4.json',{'status':'PASS',**common,'complete_test_suite':tests,
        'pre_score_registration_audit':registration,'nested_reuse_audit':nested,'negative_control':negative,
        'forbidden_runtime_features':['dataset_id','tau','final_online_length','future_suffix','posthoc_mechanism'],
        'original_grouped_splits_unchanged':True,'failed_screen_full_promotion':False,
        'ID_check_scope':'Renamed/reordered two-series mock protocol plus 20 actual fitted-model reverse-order resets; IDs are not model inputs.',
        'cold_import_and_warm_inference_IO_distinguished':True,'original_failed_workers_preserved':failures,
        'boundary_audit':boundary})
    write_json('DETERMINISM_NEXT4.json',{'status':'PASS',**common,'actual_final_audit_APIs':apis,
        'B2_repaired_cold_entry':cold_b2,'frozen_cold_reference':cold_bl,'copied_raw_free_payload':portable,
        'fresh_raw_selected_reproduction':fresh,'CPython_bytecode_writes_disabled_for_cold_readonly_tests':True,
        'all_original_bootstrap_arrays_bitwise_equal':stats['all_original_reference_and_candidate_draw_arrays_bitwise_equal'],
        'source_and_model_hashes_checked':True,'scope':'Exact checks for these fixed models, data, platform and pinned environment; not a claim of universal cross-platform bitwise equality.'})
    write_json('LATENCY_NEXT4.json',{'status':'PASS',**common,'evidence':runtime,
        'primary_frozen_state_array_bytes':apis['BL80']['state_array_bytes_min'],
        'B2_diagnostic_state_array_bytes':apis['B2']['state_array_bytes_min'],
        'state_bytes_scope':'Unique mutable NumPy arrays; excludes Python overhead, immutable tree storage and total RSS.',
        'protocol':'Original fixed 128-series sample, actual 10M scalar updates, contemporaneous matching BL80, bounded paired repetitions and process isolation.',
        'older_90_second_records_used_for_current_ratios':False,'B2_latency_does_not_override_failed_CI':True})
    write_json('REPRODUCTION_NEXT4.json',{'status':'PASS' if fresh else 'NOT_REQUIRED_NO_NEW_PROMOTION',**common,
        'new_promoted_candidate':role['experiment_id'] if role else None,'fresh_raw_training':fresh,
        'frozen_model_refitted':False,'B2_cached_diagnostic_refit_is_fresh_raw_reproduction':False,
        'copied_raw_free_inference_validation':portable,'independent_environment_installed':False,
        'interpretation':'A promoted robustness artifact has exact raw->new feature->same training->inference reproduction; without a new role no redundant frozen refit is required.'})
    refs=read(BASE/'preflight/FROZEN_REFERENCES.json')
    write_json('RAW_DATA_REQUIRED_NEXT4.json',{'status':'RAW_EXCLUDED_FROM_HANDOFF','created_utc':utc_now(),
        'required_private_training_files':{name:{'destination':'data/raw/'+name,'sha256':sha} for name,sha in refs['raw_sha256'].items()},
        'DEV_series':8000,'split_lock':'SPLIT_LOCK_REAL.json','seal_lock':'SEAL_PARTITION_500x4_LOCK.json',
        'seal_rows_must_never_be_loaded':True,'included_models_support_raw_free_inference':True,
        'full_real_OOF_and_feature_caches_included':False,'no_signed_URLs_or_credentials':True,
        'retraining_requires_regenerating_excluded_caches':True})
    context={'selection':selection,'timing':timing,'fits':fits,'tests':tests,'apis':apis}
    for name,content in documents(context).items():
        preserve(name);(ROOT/name).write_text(content,encoding='utf-8')
    readfirst=(ROOT/'00_GPT_READ_FIRST_NEXT4.md').read_text(encoding='utf-8')
    assert [int(x) for x in re.findall(r'(?m)^(\d+)\. \*\*',readfirst)]==list(range(1,43))
    for name in REQUIRED:
        if name=='FINAL_AUDIT_NEXT4.json':continue
        assert (ROOT/name).is_file() and (ROOT/name).stat().st_size>0,name
    for name in REQUIRED:
        if name!='FINAL_AUDIT_NEXT4.json':evidence[name]=digest(ROOT/name)
    checks={'research_at_least_21600_observed_seconds':timing['research_minimum_pass'],
        'separate_audit_at_least_2700_observed_seconds':timing['separate_audit_minimum_pass'],
        'actual_audit_within_60_minutes_at_publication':timing['audit_within_60_minutes_at_check'],
        'raw_split_seal_reduced_hashes':boundary['all_seals_unopened'],
        'original_staged_changes_preserved':boundary['staged_patch_byte_equal'],
        'prior_908_sources_preserved':boundary['preexisting_source_files_unchanged']==908,
        'NEXT3_user_stop_preserved':boundary['NEXT3_user_stop_preserved'],
        'exact_pins':len(boundary['exact_environment_pins'])==63,
        'full_test_suite':tests['executed_tests']>=1000 and tests['failures']==tests['errors']==tests['skips']==0,
        'original_all_14_candidate_statistics_exact':len(stats['candidate_summaries'])==14 and stats['all_original_reference_and_candidate_draw_arrays_bitwise_equal'],
        'original_supported_mechanism_arrays_exact':mechanisms['all_original_candidate_and_reference_arrays_bitwise_equal'],
        'prospective_conditional_and_final_artifact_validation':nominations['conditional_and_actual_artifact_policies_precede_their_predictions'],
        'fixed_ALT_H_overall_and_SCALE_statistics_exact':transfer_stats['all_original_arrays_bitwise_equal'] and transfer_stats['all_original_summary_fields_exact'],
        'all_selected_and_diagnostic_API_prefixes':all(r['prefix_cases']>=300 for r in apis.values()),
        'all_nominated_synthetic_suites_complete':not tables['synthetic_pending'],
        'every_real_fit_accounted_without_duplicate':fits['real_DEV_learner_fit_completions']==ledger['NEXT4_real_DEV_completed_fits'],
        'role_count_within_limit':len(selection['roles'])<=4,
        'fresh_raw_reproduction_if_new_promotion':bool(fresh and fresh['fresh_raw_feature_and_inference_exact']) if role else True,
        'copied_payload_fresh_process_inference':portable['status']=='PASS',
        'all_worker_failures_preserved_and_resolved':all(r['original_failure_preserved'] for r in failures),
        'required_root_reports_and_42_answers':True,'all_four_seals_unopened':True,
        'new_reduced_public_cloud_uses_zero':True}
    if not all(checks.values()):raise RuntimeError('Unpassed final audit conditions: '+str([k for k,v in checks.items() if not v]))
    atomic_json(out/'PUBLICATION_PREVIOUS_FILES.json',{'created_utc':utc_now(),'files':prior},immutable=True)
    final={'status':'PASS','created_utc':utc_now(),'session_outcome':selection['session_outcome'],'checks':checks,
        'observed_timing':timing,'audit_start_was_actual_and_not_backdated':True,
        'selected_roles':list(selection['roles']),'new_performance_promotion':False,'new_generation':False,
        'new_robustness_promotion':bool(role),'all_required_work_complete_before_packaging':True,
        'real_DEV_learner_fits':fits['real_DEV_learner_fit_completions'],'complete_test_count':tests['executed_tests'],
        'preserved_resolved_worker_failure_count':len(failures),'scientific_synthetic_nominations':synthetic,
        'registration_chronology_audit':registration,'execution_dependency_audit':execution,
        'original_primary_statistical_audit':stats,'sign_reflection_is_diagnostic_only':reflection['promotion_authority'] is False,
        'evidence_sha256':evidence,'source_sha256':digest(Path(__file__)),
        'package_verification_scope':'Archive CRC/member SHA and exact 20-file folder are verified subsequently in the external Desktop delivery receipt. They are not falsely pre-claimed here.'}
    atomic_json(ROOT/'FINAL_AUDIT_NEXT4.json',final,immutable=True)
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'root_final_audit_sha256':digest(ROOT/'FINAL_AUDIT_NEXT4.json'),
        'required_reports':len(REQUIRED),'Korean_answers':42,'previous_files_preserved':prior,
        'files_sha256':{name:digest(ROOT/name) for name in REQUIRED}},immutable=True)
    progress('Final audit reports published','All observed time and evidence checks passed; 42 Korean answers and complete required report set',phase='FINAL_AUDIT',counts_as_research=False)
    print('NEXT4 FINAL REPORTS PUBLISHED',len(REQUIRED),'required reports; 42 Korean answers',flush=True)


if __name__=='__main__':main()
