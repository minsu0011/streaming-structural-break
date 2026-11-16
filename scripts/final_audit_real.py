"""Evidence-based real-data final audit; never opens seal or reduced target rows."""
import argparse, ast, importlib.metadata, json, subprocess, sys, xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.features.streaming import feature_version
from src.streaming.fused import implementation_hash as fused_hash
from src.streaming.fused_ar import implementation_hash as ar_fused_hash
from src.streaming.fast_initialization import implementation_hash as initialization_hash
from src.validation.splits import FrozenSplit
from src.utils.artifacts import write_json, sha256, utc_now


def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser();modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--preflight',action='store_true')
    modes.add_argument('--prepared',action='store_true',help='Audit frozen DEV artifacts while the final reduced diagnostic is still reserved; never produces FINAL_AUDIT_REAL.json')
    parser.add_argument('--tests',default='artifacts/validation/pytest_final_real.xml');args=parser.parse_args()
    checks={};evidence={};raw=read('artifacts/audits/RAW_DATA_MANIFEST.json')
    checks['actual_raw_acquired']=raw['status']=='ACQUIRED'
    identities=[]
    for record in raw['files']:
        path=ROOT/'data/raw'/record['name']
        identities.append({'name':record['name'],'bytes':path.stat().st_size,'sha256':sha256(path),
            'matches':path.stat().st_size==record['bytes'] and sha256(path)==record['sha256']})
    checks['all_raw_bytes_and_sizes_match']=all(r['matches'] for r in identities);evidence['raw_files']=identities
    split=FrozenSplit(ROOT/'artifacts/validation');checks['frozen_group_split_valid']=True
    checks['root_assignment_copy_matches']=sha256(ROOT/'GROUP_ASSIGNMENTS.parquet')==split.digest
    checks['group_policy_unchanged']=sha256(ROOT/'artifacts/audits/GROUP_POLICY.json')==split.lock['group_policy_sha256']
    checks['no_seal_opening_recorded']=not split.lock['LOCAL_FINAL_SEAL_OPENED']
    source=read('vendor/official/SOURCE_MANIFEST.json')
    checks['official_sources_unchanged']=all(sha256(ROOT/r['path'])==r['sha256'] for r in source['sources'])
    forbidden={'tau','tau_index','target','dataset_id','online_length','has_break','y_train','y_test'}
    feature_modules=['streaming.py','historical_calibration.py','startup_calibration.py','ar_residual_input.py','ar_order_input.py','rank_gaussian.py','historical_context.py','cross_order.py','arch_input.py','historical_quantiles.py','magnitude_dependence.py']
    feature_audit={}
    for name in feature_modules:
        path=ROOT/'src/features'/name
        if not path.exists():continue
        used={node.id for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))) if isinstance(node,ast.Name)}
        feature_audit[name]={'sha256':sha256(path),'forbidden_identifiers':sorted(used&forbidden)}
    checks['inference_features_have_no_forbidden_identifiers']=all(not r['forbidden_identifiers'] for r in feature_audit.values());evidence['feature_static_audit']=feature_audit
    prefix=read('artifacts/validation/PREFIX_INVARIANCE.json')
    checks['real_prefix_feature_identity_valid']=prefix['status']=='PASS' and prefix['feature_hash']==feature_version()
    for name in ('SCORER_PARITY','SCORER_PARITY_REAL','LABEL_ALIGNMENT'):
        report=read(f'artifacts/validation/{name}.json');checks[name]=report['status']=='PASS';evidence[name]=report
    versions={}
    for line in (ROOT/'requirements-lock.txt').read_text().splitlines():
        if '==' in line:
            name,expected=line.split('==',1);versions[name]={'expected':expected,'actual':importlib.metadata.version(name)}
    checks['locked_dependencies_match']=all(r['expected']==r['actual'] for r in versions.values());evidence['versions']=versions
    try:
        git_top=subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
        own_git=Path(git_top).resolve()==ROOT.resolve()
    except (OSError,subprocess.CalledProcessError):own_git=False
    if own_git:
        tracked=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
        checks['no_raw_or_binary_data_in_git']=not any(p.startswith('data/') or p.endswith(('.parquet','.joblib','.zip')) for p in tracked)
        evidence['inventory_scope']='This exact project Git root; an unrelated parent repository is never used.'
    else:
        inventory=read('SOURCE_MANIFEST.json');tracked=[row['path'] for row in inventory['files']]
        checks['archive_inventory_excludes_raw_paths']=not any(p.startswith('data/') for p in tracked)
        evidence['inventory_scope']='Archived SOURCE_MANIFEST inventory; no Git metadata exists for this extracted project. Selected model/group binaries are intentionally included.'
    checks['credential_files_not_tracked']=not any('.crunchdao' in Path(p).parts or Path(p).name in ('token','.env','crunch.json') for p in tracked)
    tests=ROOT/args.tests
    if tests.exists():
        tree=ET.parse(tests).getroot();totals={k:sum(int(s.get(k,0)) for s in tree.iter('testsuite')) for k in ('tests','failures','errors','skipped')}
        checks['tests_pass']=totals['tests']>0 and totals['failures']==totals['errors']==0
        evidence['tests']={'path':args.tests,'sha256':sha256(tests),**totals}
    else:checks['tests_pass']=False;evidence['tests']={'path':args.tests,'status':'MISSING'}
    state=read('artifacts/real/RESEARCH_STATE.json');usage=read('REDUCED_TEST_USAGE_LEDGER.json')
    checks['research_seal_closed']=state['LOCAL_FINAL_SEAL_OPENED'] is False
    checks['only_complete_five_fold_results_ranked']=all(r['status']=='COMPLETE' and len(r['fold_scores'])==5 for r in state['results'])
    checks['reduced_quota_respected']=usage['scored_uses']<=2 and len(usage['entries'])==usage['scored_uses'] and len({e['kind'] for e in usage['entries']})==len(usage['entries'])
    exports=read('artifacts/validation/RESULT_EXPORT_AUDIT_REAL.json')
    checks['result_export_values_and_coverage_match']=exports['status']=='PASS' and exports['research_state_sha256']==sha256(ROOT/'artifacts/real/RESEARCH_STATE.json')
    checks['result_export_files_unchanged']=all(sha256(ROOT/name)==record['sha256'] for name,record in exports['files'].items())
    evidence['result_exports']=exports
    figure=read('artifacts/real/figures/FIGURE_RENDER_REPORT.json');figure_inputs=read('artifacts/real/figures/FIGURE_INPUTS.json')
    checks['dev_figure_sources_current']=figure['status']=='PASS' and figure['figure_inputs_sha256']==sha256(ROOT/'artifacts/real/figures/FIGURE_INPUTS.json') and figure['renderer_sha256']==sha256(ROOT/'scripts/render_result_figure.py') and figure_inputs['research_state_sha256']==sha256(ROOT/'artifacts/real/RESEARCH_STATE.json')
    checks['dev_figure_output_bytes_unchanged']=all(sha256(ROOT/row['path'])==row['sha256'] for row in figure['outputs'])
    closure_path=ROOT/'artifacts/real/MODEL_RESEARCH_CLOSURE.json'
    if closure_path.exists():
        closure=json.loads(closure_path.read_text());session=read('artifacts/validation/REAL_SESSION.json')
        checks['closed_research_checkpoint_unchanged']=closure['research_state_sha256']==sha256(ROOT/'artifacts/real/RESEARCH_STATE.json') and session.get('model_research_closed') is True
        checks['original_deadlines_preserved_after_closure']=all(closure[k]==session[k] for k in ('deadline_utc','new_experiment_deadline_utc'))
    if not args.preflight:
        frozen=read('DEV_CHAMPION_CANDIDATES.json');refit=read('artifacts/real/FINAL_REFIT_REAL.json')
        checks['research_stopped']=state.get('active_candidate') is None and read('artifacts/real/CONTROL.json')['pause_after_candidate']
        checks['final_candidates_frozen']=frozen['status']=='FROZEN_FOR_FINAL_DIAGNOSTICS' and frozen['split_sha256']==split.digest
        if closure_path.exists():checks['selection_matches_closed_research']=frozen['selection_state_sha256']==closure['research_state_sha256']
        checks['final_refit_and_independent_refit_pass']=refit['status']=='PASS' and refit['primary_independent_refit']['status']=='PASS' and refit['selection_sha256']==sha256(ROOT/'DEV_CHAMPION_CANDIDATES.json')
        archive_raw=read('artifacts/real/ARCHIVE_RAW_REPRODUCTION_REAL.json')
        checks['archive_raw_reproduction_selection_current']=archive_raw['status']=='PASS' and archive_raw['selection_sha256']==sha256(ROOT/'DEV_CHAMPION_CANDIDATES.json')
        checks['archive_raw_reproduction_sources_current']=all(sha256(ROOT/name)==digest for name,digest in archive_raw['checked_training_and_inference_sources'].items())
        evidence['archive_raw_reproduction']=archive_raw
        predictive=read('artifacts/real/PREDICTIVE_STATE_REPRODUCTION_REAL.json')
        checks['exact_reproduced_predictive_state_current']=predictive['status']=='PASS' and predictive['comparison_count']==18 and predictive['selection_sha256']==sha256(ROOT/'DEV_CHAMPION_CANDIDATES.json') and predictive['archive_raw_proof_sha256']==sha256(ROOT/'artifacts/real/ARCHIVE_RAW_REPRODUCTION_REAL.json') and predictive['script_sha256']==sha256(ROOT/'scripts/audit_reproduced_predictive_state.py') and all(row['exact_predictive_state_match'] for row in predictive['comparisons'])
        evidence['exact_predictive_state_reproduction']=predictive
        for role,item in frozen['candidates'].items():
            eid=item['experiment_id'];directory=ROOT/item['model_directory'];descriptor=json.loads((directory/'model.json').read_text())
            mode='fused_fast_init' if descriptor['use_fused_inference'] else 'native'
            contract=read(f'artifacts/real/contracts/{eid}_final_{mode}.json')
            checks[role+'_artifact_identity']=descriptor['model_sha256']==sha256(directory/'model.joblib') and descriptor['feature_hash']==item['feature_hash']
            checks[role+'_dev_only_fit']=descriptor['seal_rows_fitted']==0 and descriptor['development_series_fitted']==8000 and descriptor['seal_series_skipped_before_array_or_label_access']==2000
            checks[role+'_live_contract']=contract['status']=='PASS' and contract['inference_source_sha256']==sha256(ROOT/'src/streaming/inference.py') and contract['model_descriptor_sha256']==sha256(directory/'model.json')
            checks[role+'_training_source_current']=descriptor['training_source_sha256']==sha256(ROOT/'src/streaming/training.py')
            if descriptor['use_fused_inference']:
                from src.models.representations import bundle_class
                from src.streaming.prepared import source_hashes
                loaded=bundle_class(item['candidate'].get('representation','base')).load(directory/'model.joblib')
                checks[role+'_optimized_sources_current']=all(descriptor.get(k)==v for k,v in source_hashes(loaded,fast_initialization=True).items())
                cv_model=bundle_class(item['candidate'].get('representation','base')).load(ROOT/'artifacts/models/real'/eid/'fold_0.joblib')
                replay=read(f'artifacts/real/fused/{eid}_fast_init.json')
                checks[role+'_full_streaming_oof']=replay['status']=='PASS' and replay['total_rows']==4014405 and all(replay.get(k)==v for k,v in source_hashes(cv_model,fast_initialization=True).items())
            cv_refit=read(f'artifacts/real/cv_reproduction/final_selected/{eid}/RESULTS.json')
            checks[role+'_independent_cv_refit']=cv_refit['status']=='PASS' and cv_refit['total_rows']==4014405
            archived=archive_raw['candidates'][role]
            raw_cv=read(archived['cv_report']);raw_final=read(archived['final_model_report'])
            checks[role+'_archive_raw_cv_reproduction']=raw_cv['status']=='PASS' and raw_cv['total_rows']==4014405 and raw_cv['rebuilt_cache_fold_bytes_match_original'] and sha256(ROOT/archived['cv_report'])==archived['cv_report_sha256']
            checks[role+'_archive_raw_final_model_reproduction']=raw_final['status']=='PASS' and raw_final['total_rows']==4014405 and raw_final['original_model_sha256']==descriptor['model_sha256'] and raw_final['seal_rows_fitted']==0 and sha256(ROOT/archived['final_model_report'])==archived['final_model_report_sha256']
            full_score=read(f'artifacts/real/scorer_full_oof/{eid}/RESULTS.json')
            checks[role+'_full_official_scorer_parity']=full_score['status']=='PASS' and full_score['total_rows']==4014405 and full_score['max_abs_diff']<1e-12 and full_score['official_source_sha256']==sha256(ROOT/'vendor/official/scoring.py') and full_score['local_source_sha256']==sha256(ROOT/'src/scoring/ts_auc.py')
            final_stream=read(f'artifacts/real/final_streaming/{role}/RESULTS.json')
            checks[role+'_actual_full_streaming_batch_parity']=final_stream['status']=='PASS' and final_stream['total_rows']==4014405 and final_stream['different_float32_rows']==0 and final_stream['model_descriptor_sha256']==sha256(directory/'model.json') and final_stream['inference_source_sha256']==sha256(ROOT/'src/streaming/inference.py') and all(final_stream.get(k)==v for k,v in source_hashes(loaded,fast_initialization=True).items())
            if role=='PRIMARY_DEV_CANDIDATE':
                checks['actual_primary_full_dev_subprocess_socket']=final_stream.get('primary_socket_pass') is True and final_stream['subprocess_protocol_helper_sha256']==sha256(ROOT/'src/validation/subprocess_protocol.py') and final_stream['subprocess_worker_sha256']==sha256(ROOT/'scripts/official_socket_worker.py')
            timing=read(f'artifacts/real/latency/final_deployment/{eid}_{mode}.json')
            checks[role+'_actual_artifact_timing']=timing['status']=='PASS' and timing['isolated_from_research_workers'] and timing['model_sha256']==descriptor['model_sha256'] and timing['long_stream']['points']>=(10000000 if role=='PRIMARY_DEV_CANDIDATE' else 1000000)
            evidence[role]={'candidate':eid,'model_sha256':descriptor['model_sha256'],'contract':contract,'full_streaming_actual_artifact':final_stream}
        primary=frozen['PRIMARY_DEV_CANDIDATE'];reproduction=read(f'artifacts/real/cv_reproduction/final_selected/{primary}/RESULTS.json')
        checks['selected_cv_independent_refit_pass']=reproduction['status']=='PASS' and reproduction['total_rows']==4014405
        protocol_reports=[]
        for path in (ROOT/'artifacts/real/protocol').glob('*/'+primary+'/RESULTS.json'):
            protocol=json.loads(path.read_text())
            valid=(protocol.get('status')=='PASS' and protocol.get('total_rows')==4014405 and protocol.get('total_series')==8000
                and protocol.get('different_float32_rows')==0 and len(protocol.get('folds',[]))==5
                and protocol.get('inference_source_sha256')==sha256(ROOT/'src/streaming/inference.py')
                and protocol.get('runner_sha256')==sha256(ROOT/'vendor/official/runner.py')
                and protocol.get('protocol_helper_sha256')==sha256(ROOT/'src/validation/protocol.py'))
            if protocol.get('transport')=='subprocess':
                valid=valid and protocol.get('subprocess_protocol_helper_sha256')==sha256(ROOT/'src/validation/subprocess_protocol.py') and protocol.get('subprocess_worker_sha256')==sha256(ROOT/'scripts/official_socket_worker.py')
            if valid:
                from src.models.representations import bundle_class
                from src.streaming.prepared import source_hashes
                representation=frozen['candidates']['PRIMARY_DEV_CANDIDATE']['candidate'].get('representation','base')
                for row in protocol['folds']:
                    model_path=ROOT/'artifacts/models/real'/primary/f"fold_{row['fold']}.joblib"
                    model=bundle_class(representation).load(model_path)
                    valid=valid and row.get('model_sha256')==sha256(model_path) and all(row.get(k)==v for k,v in source_hashes(model,fast_initialization=True).items())
            if valid:protocol_reports.append((path,protocol))
        checks['primary_full_dev_official_socket_path']=bool(protocol_reports)
        if protocol_reports:
            path,protocol=max(protocol_reports,key=lambda item:item[1]['completed_utc'])
            evidence['primary_full_dev_socket']={'path':str(path.relative_to(ROOT)),'sha256':sha256(path),'report':protocol}
        if frozen['candidates']['PRIMARY_DEV_CANDIDATE']['candidate']['model'] in ('M3_histgb','M4_lightgbm','M5_xgboost','M7_cross_order_equal'):
            fused=read(f'artifacts/real/fused/{primary}_fast_init.json')
            representation=frozen['candidates']['PRIMARY_DEV_CANDIDATE']['candidate'].get('representation','base')
            original=bundle_class(representation).load(ROOT/'artifacts/models/real'/primary/'fold_0.joblib')
            checks['all_dev_streaming_oof_parity']=fused['status']=='PASS' and fused['total_rows']==4014405 and all(fused.get(k)==v for k,v in source_hashes(original,fast_initialization=True).items())
        for kind in (('BASELINE',) if args.prepared else ('BASELINE','PRIMARY')):
            runner=read(f'artifacts/real/CRUNCH_TEST_{kind}_REAL.json')
            checks[kind+'_official_runner_pass']=runner['status']=='PASS' and runner['official_crunch_tools_test_called'] and runner['determinism_passed'] and runner['runner_sha256']==sha256(ROOT/'vendor/official/runner.py')
            evidence[kind+'_runner']=runner
        if args.prepared:
            checks['primary_reduced_diagnostic_still_reserved']=usage['scored_uses']==1 and [entry['kind'] for entry in usage['entries']]==['baseline']
        else:
            checks['exactly_two_permitted_scored_diagnostics']=usage['scored_uses']==2 and all(e['scored_once'] for e in usage['entries'])
            primary_directory=ROOT/frozen['candidates']['PRIMARY_DEV_CANDIDATE']['model_directory']
            checks['official_primary_artifact_matches']=evidence['PRIMARY_runner']['model_descriptor_sha256']==sha256(primary_directory/'model.json')
    report={'status':('PREFLIGHT_PASS' if args.preflight else 'PREPARED_PASS' if args.prepared else 'PASS') if all(checks.values()) else 'FAIL','timestamp':utc_now(),
        'checks':checks,'evidence':evidence,'completed_candidates':len(state['results']),'main_loop_failed_candidates':len(state['failed']),
        'LOCAL_FINAL_SEAL_OPENED':False,'reduced_scored_uses':usage['scored_uses'],'scope':'Real DEV evidence audit; source/raw hashing reads bytes, not target performance. Global descriptive audit preceded the split; no seal performance was evaluated.',
        'limitations':['Adaptive DEV selection is not an independent generalization estimate.','Historical duplicate search is approximate and restricted to equal-length candidates.','Early-break discrimination remains weak.','Windows local runner and DEV loopback sockets are measured; cloud Linux portability and cloud IPC scaling are not measured.','Each selected representation computes 57 features even when learner columns are ablated; the cross-order average uses two representations.']}
    path='artifacts/validation/REAL_PREFLIGHT_AUDIT.json' if args.preflight else 'artifacts/validation/PREPARED_DEV_AUDIT_REAL.json' if args.prepared else 'artifacts/validation/FINAL_AUDIT_REAL.json'
    write_json(ROOT/path,report);print(report['status'],len(checks),'checks; failed:',[k for k,v in checks.items() if not v],flush=True)
    if not all(checks.values()):raise SystemExit(1)


if __name__=='__main__':main()
