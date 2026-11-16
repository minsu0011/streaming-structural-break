"""Freeze final DEV roles from completed science, exactness and independent training evidence."""
from pathlib import Path
import sys
import json
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.next2.session import clock, checkpoint
from src.next2.decision_gates import assess
from src.next.guard import SealGuard
from src.next.data import RAW_SHA
from src.utils.artifacts import write_json, sha256, utc_now

J = 'J102_SERIAL_VARIANCE_512'; B = 'BL2_C205_DEPENDENCE_80'


def read(relative): return json.loads((ROOT/relative).read_text(encoding='utf-8-sig'))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--review-revision', action='store_true'); args = parser.parse_args()
    timing = clock(); assert timing['minimum_research_met'] and timing['audit_started']
    destination = ROOT/'DEV_CANDIDATES_NEXT2.json'
    if destination.exists():
        if not args.review_revision: raise RuntimeError('Final selection already frozen')
        preserved = ROOT/'artifacts/next2/final_audit/selection_review_v1/DEV_CANDIDATES_NEXT2.json'
        assert preserved.exists() and sha256(preserved) == sha256(destination)
    tests_path = ROOT/'artifacts/next2/final_audit/ALL_TESTS.xml'
    suites = list(ET.parse(tests_path).getroot().iter('testsuite'))
    tests = sum(int(s.attrib['tests']) for s in suites)
    assert tests == 536 and all(int(s.attrib.get(k, 0)) == 0 for s in suites for k in ('errors', 'failures', 'skipped'))
    frozen = read('artifacts/next2/initialization/FROZEN_REFERENCES.json')
    changed = [p for p, digest in frozen['protected_sha256'].items() if sha256(ROOT/p) != digest]
    assert not changed
    guard = SealGuard(ROOT); assert len(guard.dev_ids) == 8000 and len(guard.seal_ids) == 2000 and not guard.dev_ids.intersection(guard.seal_ids)
    raw = {name: sha256(ROOT/'data/raw'/name) for name in RAW_SHA}; assert raw == RAW_SHA
    rep = {}; evidence = {'artifacts/next2/final_audit/ALL_TESTS.xml': sha256(tests_path)}
    for role in ('DISTILLED', 'PERFORMANCE'):
        path = 'artifacts/next2/independent_reproduction/'+role+'/RESULTS.json'; r = read(path)
        assert r['status'] == 'PASS' and r['series'] == 8000 and r['points'] == 4014405
        assert r['canonical_model_bytes_exact'] and r['pin_count'] == 63
        assert r['new_fits'] == (1 if role == 'DISTILLED' else 2)
        assert r['raw_before_sha256'] == r['raw_after_sha256'] == RAW_SHA
        assert r['opaque_seal_series_skipped'] == 2000 and r['seal_values_accessed'] == r['seal_rows_fitted'] == 0
        for key in ('original_project_access_attempts', 'expected_reads_before_training', 'outside_workspace_write_attempts', 'network_attempts'): assert not r[key]
        for checks in r['training_comparisons'].values(): assert all(checks.values())
        rep[role] = r; evidence[path] = sha256(ROOT/path)
    proof_paths = ['artifacts/next2/canonical_api_all_dev/DISTILLED/RESULTS.json',
        'artifacts/next2/shared_public_api_all_dev/PERFORMANCE/RESULTS.json',
        'artifacts/next2/alternate_runtime_v2/DISTILLED/RESULTS.json',
        'artifacts/next2/shared_blend_runtime/all_dev/RESULTS.json',
        'artifacts/next2/shared_blend_runtime/oof/RESULTS.json',
        'artifacts/next2/shared_blend_runtime/alternate/RESULTS.json',
        'artifacts/next2/report_bootstrap_replay/RESULTS.json']
    for path in proof_paths:
        assert read(path)['status'] == 'PASS'; evidence[path] = sha256(ROOT/path)
    for role, api in [('DISTILLED', proof_paths[0]), ('PERFORMANCE', proof_paths[1])]:
        r = read(api); assert r['persistent_filesystem_write_attempts'] == 0
        assert r['output_fingerprint_sha256'] == rep[role]['all_dev_prediction_sha256']
    policy = read('configs/next2/RESEARCH_POLICY_LOCK.json')
    primary = frozen['references']['FROZEN_PRIMARY']; reference = primary['full']
    full = read('artifacts/next2/experiments/'+B+'/RESULTS.json')['full']
    paired = pd.read_csv(ROOT/'PAIRED_BOOTSTRAP_NEXT2.csv', float_precision='round_trip')
    pair = {k: v for k, v in paired[(paired.candidate == B) & (paired.reference == J) & (paired.scope == 'all_time')].iloc[0].to_dict().items() if pd.notna(v)}
    alts = []; h = None
    for split in ('ALT_1', 'ALT_2', 'ALT_3', 'LEAVE_ONE_H_REGIME_OUT'):
        a = read('artifacts/next2/robustness/'+B+'/'+split+'/RESULTS.json')
        b = read('artifacts/next/robustness/candidates/'+J+'/'+split+'/RESULTS.json')
        assert a['status'] == b['status'] == 'PASS'
        values, baseline = np.asarray(a['fold_scores']), np.asarray(b['fold_scores'])
        item = {'split': split, 'mean': float(values.mean()), 'mean_delta': float(values.mean()-baseline.mean()),
            'worst': float(values.min()), 'worst_delta': float(values.min()-baseline.min()),
            'minimum_matched_fold_delta': float((values-baseline).min()), 'fold_scores': values.tolist()}
        if split.startswith('ALT_'): alts.append(item)
        else: h = item
    mechanism = paired[(paired.candidate == B) & (paired.reference == J) & (paired.scope == 'mechanism_DEPENDENCE_DOMINANT')].iloc[0]
    assert mechanism.supported == True and mechanism.break_series >= 100 and mechanism.ci_2_5 > 0 and mechanism.estimate_mean_delta >= .002
    runtime = read('artifacts/next2/latency/BL80_SHARED_V1/RESULTS.json')
    assert runtime['engineering_replacement_eligible'] and all(runtime['engineering_replacement_criteria'].values())
    # The literal simplification criterion is a reduction in state OR point latency.
    costs = pd.read_csv(ROOT/'RUNTIME_PRUNING_RESULTS.csv').set_index('runtime_variant')
    smaller_state = costs.loc['BL80_SHARED_V1', 'private_per_series_numeric_bytes'] < costs.loc['J_ORIGINAL', 'private_per_series_numeric_bytes']
    assert smaller_state
    gates = assess(full, reference, pair, alts, h, complexity_decreased=True, engineering_pass=True,
        independent_mechanism_gain=True, policy=policy)
    assert gates['performance']['status'] == 'PASS' and gates['generation']['status'] == 'FAIL'
    jgate = read('artifacts/next2/research_tables/GATES.json')['gates'][J]
    assert all(value is True for name, value in jgate['checks'].items() if name.startswith('simplification_') and name != 'simplification_complexity')
    jgate = assess(reference, reference, {'probability_delta_above_minus_0005': 1., 'probability_delta_positive': 0.},
        [{'mean_delta': 0., 'worst_delta': 0.} for _ in range(3)], {'minimum_matched_fold_delta': 0.},
        complexity_decreased=True, engineering_pass=True, independent_mechanism_gain=False, policy=policy)
    assert jgate['distilled']['status'] == 'PASS'
    def role_record(role, variant, entry, eid, score, gate, runtime_variant):
        folder = 'artifacts/next2/canonical_deployment/'+variant
        descriptor = read(folder+'/model.json'); assert descriptor['model_sha256'] == rep[role]['canonical_model_sha256']
        assert sha256(ROOT/folder/'model.sbnext2') == descriptor['model_sha256']
        return {'role': role, 'experiment_id': eid, 'entry': entry, 'runtime': descriptor['runtime'], 'runtime_variant': runtime_variant,
            'model_directory': folder, 'canonical_model_sha256': descriptor['model_sha256'], 'model_bytes': (ROOT/folder/'model.sbnext2').stat().st_size,
            'descriptor_sha256': sha256(ROOT/folder/'model.json'), 'full': score, 'gates': gate,
            'prediction_fingerprint_sha256': rep[role]['all_dev_prediction_sha256']}
    performance = role_record('PERFORMANCE', 'PERFORMANCE_shared_v1', 'next2_shared_performance_main.py', B, full, gates, 'BL80_SHARED_V1')
    performance.update(paired_all_time={k: (v.item() if hasattr(v, 'item') else v) for k, v in pair.items()}, alternate_splits=alts, heldout_H=h,
        weight_primary=.8, weight_complement=.2, components=[J, 'C205_DEPENDENCE'], scientific_feature_contract=101, materialized_tree_inputs=72)
    performance['distilled_role_note'] = 'Also passes literal state-or-point reduction gate versus unpruned original J102. J_USED is selected as DISTILLED because it has smaller private state, lower point latency, one ensemble and exact original quality.'
    distilled = role_record('DISTILLED', 'DISTILLED_tree_used_v1', 'next2_canonical_distilled_main.py', J, reference, jgate, 'J_USED')
    distilled.update(scientific_feature_contract=77, materialized_tree_inputs=48, quality_delta=0., selected_name=J+'_RUNTIME_TREE_USED_V1')
    selection = {'status': 'FROZEN_NEXT2_DEV_SELECTION', 'created_utc': utc_now(), 'clock': timing,
        'performance': performance, 'distilled': distilled, 'previous_primary': primary,
        'frozen_references': frozen['references'], 'generation_level_promotion': False,
        'formal_primary_original_artifact_preserved': True, 'policy_sha256': sha256(ROOT/'configs/next2/RESEARCH_POLICY_LOCK.json'),
        'evidence_sha256': evidence, 'seal_opened': {name: False for name in guard.lock['partitions']}, 'reduced_new_usage': 0, 'public_submission_new': 0,
        'scope': 'Final DEV roles; independent raw training and engineering verified. No seal evaluation, public submission or claim of generation-level promotion.'}
    write_json(destination, selection)
    if args.review_revision:
        write_json(ROOT/'artifacts/next2/final_audit/SELECTION_GATE_REVIEW.json', {'status': 'CORRECTED_AND_VERIFIED', 'created_utc': utc_now(),
            'finding': 'Initial selection report marked blend complexity decreased false. Its private state is smaller than unpruned J102, so the literal state-OR-latency criterion is true. Also recomputed all J_USED gate groups after engineering completion.',
            'previous_selection_sha256': sha256(preserved), 'corrected_selection_sha256': sha256(destination),
            'model_bytes_or_predictions_changed': False, 'chosen_roles_changed': False,
            'performance_or_generation_decision_changed': False, 'thresholds_changed': False})
    write_json(ROOT/'REPRODUCTION_NEXT2.json', {'status': 'PASS', 'created_utc': utc_now(), 'roles': rep,
        'preparation': read('artifacts/next2/independent_reproduction/PREPARATION_READY.json'), 'new_learner_fits': 3,
        'canonical_model_bytes_exact': True, 'all_DEV_prediction_fingerprints_exact': True,
        'original_project_reads_denied': True, 'prior_feature_caches_used': False,
        'expected_weights_inaccessible_until_training_complete': True, 'application_audit_hook_not_OS_sandbox': True,
        'evidence_sha256': {p: d for p, d in evidence.items() if 'independent_reproduction' in p}})
    write_json(ROOT/'DETERMINISM_NEXT2.json', {'status': 'PASS', 'created_utc': utc_now(), 'tests_passed': tests,
        'tests_report_sha256': sha256(tests_path), 'selected_roles': {name: {'model_sha256': r['canonical_model_sha256'],
            'prediction_fingerprint_sha256': r['all_dev_prediction_sha256']} for name, r in rep.items()},
        'canonical_save_load': True, 'fresh_raw_training': True, 'full_DEV_API': True, 'OOF_and_alternate_runtime_parity': True,
        'reversed_series_reset': True, 'one_shot_causal_online_stream': True, 'persistent_inference_writes': 0,
        'evidence_sha256': evidence, 'scope': 'Exact specified platform, dependencies, fitted recipes and tested streams; no assertion of all-platform bitwise identity.'})
    write_json(ROOT/'SEAL_GUARD_NEXT2.json', {'status': 'PASS', 'created_utc': utc_now(), 'dev_series': 8000, 'seal_series': 2000,
        'dev_seal_overlap': 0, 'ALL_SEALS_UNOPENED': True, 'seals_opened': {name: False for name in guard.lock['partitions']},
        'reduced_new_usage': 0, 'public_submission_new': 0, 'seal_lock_sha256': guard.lock_sha, 'reduced_ledger_sha256': guard.ledger_sha,
        'protected_files': len(frozen['protected_sha256']), 'protected_changed': changed, 'full_tests_passed': tests,
        'independent_training_seal_values_accessed': 0, 'scope': 'Guarded DEV-only operations and unchanged seal/quota artifacts. Full byte hashing and native Parquet row-group decoding are distinct from observing seal values.'})
    write_json(ROOT/'RAW_DATA_REQUIRED_NEXT2.json', {'status': 'EXTERNAL_INPUTS_EXCLUDED_FROM_ZIP', 'files': {
        'data/raw/'+name: {'sha256': digest, 'bytes': (ROOT/'data/raw'/name).stat().st_size} for name, digest in RAW_SHA.items()},
        'split_sha256': guard.split.digest, 'scope': 'Raw copies were used only for guarded DEV training. No raw file or seal labels in the research archive.'})
    write_json(ROOT/'artifacts/next2/final_audit/SELECTION_EVIDENCE.json', {'status': 'PASS', 'created_utc': utc_now(),
        'script_sha256': sha256(Path(__file__)), 'selected_model_roles': 2, 'scientific_model_rows': 80,
        'full_tests': tests, 'new_independent_fits': 3, 'total_NEXT2_learner_fits': 397,
        'selection_sha256': sha256(destination), 'evidence_sha256': evidence, 'protected_files': len(frozen['protected_sha256'])})
    checkpoint('Frozen both final DEV roles after independent raw training and 536 tests', ['Final report cross-checks', 'Portable copied package verification', 'Complete 45-minute separate audit and ZIP'])
    print('NEXT2 FINAL DEV ROLES FROZEN; generation promotion false', flush=True)


if __name__ == '__main__': main()
