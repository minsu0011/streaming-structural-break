"""Join actual, hash-verified engineering evidence without assigning final roles."""
from pathlib import Path
import argparse
import json
import math
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.guard import SealGuard
from src.next.candidate_io import validate_sources, model_class
from src.next.references import verify_reference_sources
from src.utils.artifacts import sha256, write_json, utc_now


def require(value, reason):
    if not value:
        raise RuntimeError(reason)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--reproduction-report', required=True)
    parser.add_argument('--refit-tag', default='research_v1')
    parser.add_argument('--cv-contract-tag', default='v1')
    parser.add_argument('--latency-tag', default='all_dev_v1')
    args = parser.parse_args()
    out = ROOT/'artifacts/next/final_engineering'/args.candidate
    require(not (out/'RESULTS.json').exists(), 'Keep completed engineering receipts; do not overwrite them')
    guard = SealGuard(ROOT)
    guard.check_state()
    verify_reference_sources(ROOT)
    evidence = {}
    def pin(path, expected=None):
        path = ROOT/path
        actual = sha256(path)
        require(expected is None or actual == expected, 'Engineering input bytes changed: '+str(path.relative_to(ROOT)))
        evidence[path.relative_to(ROOT).as_posix()] = actual
        return actual
    def read(path, status='PASS'):
        path = ROOT/path
        pin(path)
        result = json.loads(path.read_text(encoding='utf-8'))
        require(status is None or result.get('status') == status, 'Incomplete engineering receipt: '+str(path.relative_to(ROOT)))
        return result
    def source_map(mapping):
        for path, digest in mapping.items():
            pin(path, digest)
    original_dir = ROOT/'artifacts/next/experiments'/args.candidate
    original = read(original_dir/'RESULTS.json', status=None)
    require(original.get('full', {}).get('status') == 'COMPLETE' and original['status'] not in ('BUG', 'SCREEN_FAIL'), 'Complete valid primary CV required')
    candidate = original['candidate']
    require(candidate['experiment_id'] == args.candidate, 'Candidate ID differs')
    validate_sources(candidate)
    kind = candidate.get('extension', {}).get('kind')
    require(kind in ('log_energy', 'conditional_variance', 'serial_variance'), 'Unreviewed engineering adapter')
    if kind == 'serial_variance':
        from src.next.serial_inference import implementation_hash as infer_hash, load_artifact
        from src.next.serial_variance_runtime import source_hash as runtime_hash
        from src.next.registered_final_fitting import source_hash as fitting_hash
        variant, entry = 'serial_variance_fast_h', 'next_serial_main.py'
        runtime_driver, api_driver, helper = 'verify_next_serial_runtime.py', 'verify_next_serial_oof.py', 'verify_next_serial_contract.py'
        latency_driver, latency_policy = 'benchmark_next_serial_all_dev.py', 'configs/next_serial_all_dev_latency_policy.json'
    elif kind == 'log_energy':
        from src.next.registered_inference import implementation_hash as infer_hash, load_artifact
        from src.next.log_energy_runtime import source_hash as runtime_hash
        from src.next.registered_final_fitting import source_hash as fitting_hash
        variant, entry = 'log_energy_fast_h', 'next_registered_main.py'
        runtime_driver, api_driver, helper = 'verify_next_log_energy_runtime.py', 'verify_next_registered_oof.py', 'verify_next_registered_contract.py'
        latency_driver, latency_policy = 'benchmark_next_registered_all_dev.py', 'configs/next_registered_all_dev_latency_policy.json'
    else:
        from src.next.inference import implementation_hash as infer_hash, load_artifact
        from src.next.fast_variance_initialization import source_hash as runtime_hash
        from src.next.final_fitting import source_hash as fitting_hash
        variant, entry = 'pruned_variance_fast_h', 'next_readonly_main.py'
        runtime_driver, api_driver, helper = 'verify_next_fast_variance.py', 'verify_next_inference_oof.py', 'verify_next_inference_contract.py'
        latency_driver, latency_policy = 'benchmark_next_all_dev.py', 'configs/next_all_dev_latency_policy.json'
    fold_sources = {'policy_sha256': pin(original_dir/'full/POLICY.json')}
    for fold in range(5):
        record = read(original_dir/f'full/fold_{fold}.json', status=None)
        for name, digest in record['files_sha256'].items():
            fold_sources[name] = pin(original_dir/'full'/name, digest)
        # Bundle load validates the frozen feature, predictor and registration
        # source identities as well as the outer file hash checked above.
        model_class(candidate).load(original_dir/f'full/fold_{fold}.joblib')
    reproduction_path = ROOT/args.reproduction_report
    reproduction = read(reproduction_path)
    require(reproduction.get('seal_rows') == 0, 'Reproduction crossed the seal boundary')
    for relative, digest in reproduction['portable_receipts_sha256'].items():
        pin(reproduction_path.parent/relative, digest)
    matches = [r for r in reproduction['candidates'] if r['candidate'] == args.candidate]
    require(len(matches) == 1, 'Exact candidate reproduction receipt required')
    reproduced = matches[0]
    require(reproduced['status'] == 'PASS' and reproduced['cv_points'] == 4014405 and len(reproduced['folds']) == 5, 'Independent primary OOF coverage incomplete')
    for fold, record in enumerate(reproduced['folds']):
        require(record['fold'] == fold and record['status'] == 'PASS' and record['model_bytes_equal'] and record['all_float32_predictions_bitwise_equal_by_sha256'], 'Independent fold/model parity failure')
        for name, digest in record['files_sha256'].items():
            require(fold_sources[name] == digest, 'Independent OOF payload does not match current original CV')
    checks = {'full_primary_oof_reproduction': True}
    runtime = read(f'artifacts/next/runtime_parity/{args.candidate}/{variant}/RESULTS.json')
    require(runtime['series'] == 8000 and runtime['points'] == 4014405 and runtime['all_float32_predictions_bitwise_equal'], 'Full runtime coverage or parity failure')
    require(runtime['identity']['candidate'] == candidate and runtime['identity']['oof_sources'] == fold_sources and runtime['identity']['runtime_sha256'] == runtime_hash(), 'Runtime model/source identity differs')
    pin('scripts/'+runtime_driver, runtime['identity']['driver_sha256'])
    pin('src/next/runtime_verification.py', runtime['identity']['verification_source_sha256'])
    pin('src/next/candidate_io.py', runtime['identity']['candidate_io_sha256'])
    require(len(runtime['folds']) == 5, 'Runtime verification lacks five folds')
    for fold, row in enumerate(runtime['folds']):
        require(row['fold'] == fold and row['status'] == 'PASS' and row['fixed_memory'] and row['all_float32_predictions_bitwise_equal'] and row['maximum_absolute_difference'] == 0., 'Runtime fold parity/fixed-memory failure')
        require(read(f'artifacts/next/runtime_parity/{args.candidate}/{variant}/fold_{fold}.json') == row, 'Runtime fold receipt differs from summary')
    checks['full_runtime_parity'] = True
    api = read(f'artifacts/next/inference_oof/{args.candidate}/RESULTS.json')
    require(api['series'] == 8000 and api['points'] == 4014405 and api['all_float32_predictions_bitwise_equal'] and api['inference_file_writes'] == 0, 'Full original OOF API coverage/parity failure')
    require(api['identity']['candidate'] == candidate and api['identity']['oof_sources'] == fold_sources and api['identity']['inference_sha256'] == infer_hash() and api['identity']['runtime_sha256'] == runtime_hash(), 'Full OOF API source/model identity differs')
    pin('scripts/'+api_driver, api['identity']['script_sha256'])
    pin('scripts/'+helper, api['identity']['guard_helper_sha256'])
    for fold, record in enumerate(api['folds']):
        require(record['fold'] == fold and record['status'] == 'PASS' and record['single_pass_and_no_read_ahead'] and record['readiness_consumes_zero_inputs'], 'Full OOF API single-pass contract failure')
    checks['full_inference_oof_parity'] = True
    directory = ROOT/'artifacts/next/refits'/args.candidate/('cache_'+args.refit_tag)
    model, descriptor = load_artifact(directory)
    model_sha, descriptor_sha = pin(directory/'model.joblib'), pin(directory/'model.json')
    refit = read(directory/'RESULTS.json')
    require(refit['model_sha256'] == model_sha == reproduced['all_dev_match']['model_sha256'] and refit['descriptor_sha256'] == descriptor_sha, 'Actual all-DEV model differs from independent reproduction')
    require(refit['training'] == reproduced['all_dev_training'] and refit['training']['source_sha256'] == fitting_hash(), 'Independent all-DEV feature/target/order or fitter identity differs')
    require(refit['all_dev_prediction_sha256'] == reproduced['all_dev_match']['all_dev_prediction_sha256'] and refit['fingerprint_points'] == 4014405, 'Independent all-DEV prediction fingerprint differs')
    checks['independent_raw_all_dev_match'] = True
    cv_contract = read(f'artifacts/next/inference_contract/{args.candidate}/{args.cv_contract_tag}.json')
    require(cv_contract['model_sha256'] == fold_sources['fold_0.joblib'] and cv_contract['modified_future_suffix_cases'] >= 300 and cv_contract['file_writes_or_mutations'] == 0, 'CV future-suffix contract incomplete')
    require(cv_contract['outer_and_inner_iterators_single_pass'] and cv_contract['readiness_consumes_zero_outer_or_online_items'] and cv_contract['future_nan_suffix_never_consumed'], 'CV one-pass/future-read contract failed')
    require(cv_contract['inference_sha256'] == infer_hash(), 'CV contract inference source changed')
    pin('scripts/'+helper, cv_contract['script_sha256'])
    actual_dir = ROOT/'artifacts/next/all_dev_artifact_contract'/args.candidate/args.refit_tag
    actual = read(actual_dir/'RESULTS.json')
    for relative, digest in actual['files_sha256'].items():
        pin(actual_dir/relative, digest)
    source_map(actual['identity']['source_sha256'])
    require(actual['identity']['model_sha256'] == model_sha and actual['identity']['descriptor_sha256'] == descriptor_sha and actual['identity']['inference_sha256'] == infer_hash() and actual['identity']['runtime_sha256'] == runtime_hash(), 'Actual-artifact contract identity differs')
    full_api, suffix, cold = (read(actual_dir/name) for name in ('FULL_API.json', 'SUFFIX_CONTRACT.json', 'COLD_IMPORT.json'))
    require(full_api['points'] == 4014405 and full_api['series'] == 8000 and full_api['output_sha256'] == refit['all_dev_prediction_sha256'] and full_api['inference_audit']['filesystem_mutation_attempts'] == 0, 'Actual-artifact full API parity failed')
    require(suffix['series'] == 32 and suffix['modified_future_suffix_cases'] >= 300 and suffix['single_pass_and_no_read_ahead'] and suffix['zero_read_readiness'] and suffix['future_nan_suffix_not_consumed'] and suffix['inference_audit']['filesystem_mutation_attempts'] == 0, 'Actual-artifact future/iterator contract failed')
    checks['single_pass_future_suffix_contract'] = True
    require(cold['entry'] == Path(entry).stem and cold['model_sha256'] == model_sha and cold['series'] == 32 and cold['points'] > 10000 and cold['cold_import_included_in_write_guard'] and cold['first_jit_readiness_included_in_write_guard'] and cold['bitwise_expected_prediction_parity'] and not cold['audit']['filesystem_mutation_attempts'], 'Actual-artifact cold import/JIT/write-denial failed')
    from src.next.readonly_bootstrap import source_hash as bootstrap_hash
    require(cold['bootstrap_sha256'] == bootstrap_hash(), 'Cold-import bootstrap source changed')
    checks['cold_import_readonly'] = True
    wire = read(f'artifacts/next/official_runner/{args.candidate}/all_dev_v1/RESULTS.json')
    require(wire['model_sha256'] == model_sha and wire['descriptor_sha256'] == descriptor_sha and wire['entrypoint'] == entry, 'Official local wire used a different artifact')
    pin(entry, wire['entry_sha256'])
    source_map(wire['source_sha256'])
    require(wire['official_run_function'] and wire['official_execute_function'] and wire['official_prediction_check'] and wire['indices_match'] and wire['prediction_dtype'] == 'float32' and wire['rows'] == 2533 and wire['official_10_percent_determinism'] == [True] and wire['direct_vs_wire_max_abs_diff'] == 0., 'Official local protocol did not pass')
    require(all(row['exit_code'] == 0 for row in wire['subprocess_calls']) and wire['authenticated_sdk_test_called'] is False and wire['competition_evaluation_data_used'] is False, 'Official local execution/evaluation boundary failure')
    checks['official_local_wire'] = True
    latency = read(f'artifacts/next/latency/{args.latency_tag}/{args.candidate}.json')
    baseline = read(f'artifacts/next/latency/{args.latency_tag}/PRIMARY.json')
    require(latency['identity']['model_sha256'] == model_sha and latency['identity']['model_kind'] == baseline['identity']['model_kind'] == 'ALL_DEV_REFIT_ARTIFACT', 'Actual-artifact latency model identity differs')
    pin('scripts/'+latency_driver, latency['identity']['driver_sha256'])
    pin(latency_policy, latency['identity']['policy_sha256'])
    protocol = read(latency_policy, status='LOCKED')
    base_protocol = read('configs/next_all_dev_latency_policy.json', status='LOCKED')
    pin('configs/next_all_dev_latency_policy.json', baseline['identity']['policy_sha256'])
    pin('scripts/benchmark_next_all_dev.py', baseline['identity']['driver_sha256'])
    for key in ('series_ids', 'continuous_points', 'block_points', 'measured_checkpoint_points', 'baseline_model_sha256', 'baseline_descriptor_sha256'):
        require(protocol[key] == base_protocol[key], 'Candidate and baseline timing protocols differ: '+key)
    require(baseline['identity']['model_sha256'] == base_protocol['baseline_model_sha256'], 'Baseline latency used a different all-DEV model')
    pin(ROOT/base_protocol['baseline_model_directory']/'model.joblib', base_protocol['baseline_model_sha256'])
    pin(ROOT/base_protocol['baseline_model_directory']/'model.json', base_protocol['baseline_descriptor_sha256'])
    for report in (latency, baseline):
        require(report['real_history_initialization']['series'] == 128 and report['point_distribution_us']['samples'] == 65541, 'Fixed real timing sample changed')
        require(all(math.isfinite(report['point_distribution_us'][key]) and report['point_distribution_us'][key] > 0 for key in ('median', 'p95', 'p99', 'mean')), 'Invalid scalar timing statistic')
    require(latency['isolated_from_workspace_research_workers'] and baseline['isolated_from_workspace_research_workers'] and latency['all_sampled_predictions_bitwise_equal_to_scalar_bundle'], 'Actual-artifact timing was not isolated/verified')
    require(latency['continuous_stream']['points'] == 10000000 and latency['continuous_stream']['state_array_bytes_before'] == latency['continuous_stream']['state_array_bytes_after'], 'Actual-artifact 10M-point/fixed-memory check incomplete')
    checks['final_artifact_latency'] = True
    # All relevant inference/model source contracts above were checked against
    # current bytes. Pin the stable policy/data-boundary files as well.
    for relative in ('ROBUSTNESS_SPLIT_LOCK.json', 'SEAL_PARTITION_500x4_LOCK.json', 'REDUCED_TEST_USAGE_LEDGER.json',
                     'configs/next_final_role_policy.json', 'src/next/guard.py', 'src/next/data.py', 'src/next/readonly_bootstrap.py'):
        pin(relative)
    checks['frozen_source_identity'] = True
    require(len(checks) == 9 and all(checks.values()), 'Incomplete engineering decision')
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(), 'candidate': candidate,
        'checks': checks, 'model_sha256': model_sha, 'descriptor_sha256': descriptor_sha,
        'entrypoint': entry, 'runtime': variant, 'evidence_files_sha256': evidence,
        'point_latency_ratio': latency['point_distribution_us']['median']/baseline['point_distribution_us']['median'],
        'amortized_latency_ratio': latency['real_history_initialization']['amortized_us_per_point']/baseline['real_history_initialization']['amortized_us_per_point'],
        'auditor_sha256': sha256(Path(__file__)), 'seal_rows': 0, 'reduced_new_uses': 0,
        'final_role_assigned': False, 'research_closed': False,
        'scope': 'Engineering qualification of exact candidate/artifact bytes. Statistical promotion and final-role policies remain separate; no final handoff or seal opening is authorized by this receipt.'})
    print('ENGINEERING QUALIFIED', args.candidate, '9/9', flush=True)


if __name__ == '__main__':
    main()
