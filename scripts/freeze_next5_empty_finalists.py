"""Close this finite menu under the original gate; preserve unopened Shadow."""
from pathlib import Path
import csv
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next5.artifacts import BASE, atomic_json, read_json, digest, utc_now, progress
from src.next5.guard import ResearchGuard
from src.next5.report_tables import ScalarEvidence, primary_tables


def main():
    session = read_json(BASE / 'SESSION.json')
    account = read_json(BASE / 'runtime/ACCOUNTING.json')
    assert session['status'] == 'ACTIVE_RESEARCH' and not session.get('audit_started_utc')
    assert not session.get('shadow_opened')
    assert 0 <= time.time() - account['wall_timestamp'] <= 60
    assert account['productive_research_seconds'] >= 36000, 'The observed research minimum is incomplete'
    guard = ResearchGuard()
    guard.check_state()
    shadow = read_json(BASE / 'shadow/STATE.json')
    assert shadow['status'] == 'CLOSED' and not shadow['labels_loaded'] and shadow['evaluations_completed'] == 0
    evidence = ScalarEvidence()
    candidates = primary_tables(evidence)
    assert len(candidates) == 20 and sum(r['full_completed'] for r in candidates) == 19
    full = sorted((r for r in candidates if r['full_completed']), key=lambda r: (-r['mean_delta'], r['candidate']))
    # This script is explicitly for the observed no-eligible-candidate outcome.
    # It cannot force an empty selection if a later or changed result is eligible.
    assert all(r['primary_gate_failure_reasons'] for r in full)
    assert all(r['mean_delta'] < r['primary_mean_delta_gate'] for r in full)
    for relative, expected in (
        ('synthetic/fitted_summary/complete_v1/RESULTS.json', 'PASS'),
        ('synthetic/fitted_summary/complete_v1/COHORT_PROVENANCE_REPAIR.json', 'PASS'),
        ('latency/fixed_v1/RESULTS.json', 'PASS'),
        ('retained_artifact_validation/research_v1/RESULTS.json', 'PASS'),
        ('impulse_decomposition/complete_v1/RESULTS.json', 'PASS'),
        ('scale_direction_response/complete_v1/RESULTS.json', 'PASS'),
        ('null_drift/fixed_v1/RESULTS.json', 'PASS'),
        ('maintenance/READONLY_IMPORT_REPAIR/RESOLVED.json', 'PASS'),
    ):
        assert evidence.json(BASE / relative)['status'] == expected
    policy = evidence.json(ROOT / 'configs/next5/RESEARCH_POLICY_LOCK.json')
    assert policy['shadow_confirmation']['all_roles_require_common_research_finalist_gate']
    now = utc_now()
    manifest = {'created_utc': now, 'status': 'FROZEN_NEXT5_FINALISTS', 'finalists': [],
        'finalist_count': 0, 'unused_shadow_slots': 3, 'decision': 'NO_ELIGIBLE_FINALIST_KEEP_SHADOW_CLOSED',
        'reason': 'All 19 full candidates fail the original mean delta >= 0.0012 requirement; M2 failed screen. No rounding or robustness-only exception.',
        'research_productive_seconds_at_lock': account['productive_research_seconds'],
        'frozen_common_gate': policy['performance_gate'],
        'median_fold_interpretation': {
            'frozen_implementation': 'median(candidate fold scores) - median(reference fold scores)',
            'matched_comparison_reported': 'median(candidate fold score - matched reference fold score)',
            'bootstrap_median_interval_targets': 'Difference of score medians, not median of matched differences',
            'selection_effect': 'None: all 19 full candidates fail the original mean improvement floor.'},
        'worst_fold_interpretation': {
            'frozen_implementation': 'min(candidate fold scores) - min(reference fold scores)',
            'strict_matched_comparison_reported': 'min(candidate fold score - matched reference fold score)',
            'difference_from_per_fold_degradation_reading': True,
            'selection_effect': 'None: all 19 full candidates already fail the original mean improvement floor under either interpretation.'},
        'candidate_admission': [{k: r[k] for k in ('candidate', 'family', 'full_completed',
            'score_scope', 'mean_delta', 'mean_delta_gate_margin', 'minimum_fold_delta',
            'minimum_matched_fold_delta_floor_pass', 'median_delta', 'median_matched_fold_delta',
            'median_matched_fold_delta_floor_pass', 'primary_gate_failure_reasons')} for r in candidates],
        'shadow_opened': False, 'shadow_labels_loaded': False, 'shadow_evaluations': 0,
        'shadow_score': None, 'shadow_confirmation': 'NOT_EVALUATED_NO_ELIGIBLE_CANDIDATE',
        'new_fits_after_lock_permitted': False, 'evidence_sha256': evidence.inputs,
        'source_sha256': digest(Path(__file__))}
    atomic_json(BASE / 'shadow/FINALIST_MANIFEST_NEXT5.json', manifest, immutable=True)
    atomic_json(ROOT / 'FINALIST_MANIFEST_NEXT5.json', manifest, immutable=True)
    refs_path = BASE / 'preflight/FROZEN_REFERENCES.json'
    refs = read_json(refs_path)
    roles = {}
    for name in ('FROZEN_PRIMARY', 'NEXT4_ROBUSTNESS'):
        old = refs['roles'][name]
        assert digest(ROOT / old['model_file']) == old['model_sha256']
        roles[name] = {k: old[k] for k in ('experiment_id', 'entry', 'model_directory', 'model_file', 'model_sha256', 'model_bytes')}
        roles[name].update(retained_existing_role=True, new_NEXT5_fit=False,
            original_DEV_score_is_historical=True, original_primary=old['original_primary'],
            NEXT5_fresh_raw_training_reproduction_required=False,
            scope='Original primary retained' if name == 'FROZEN_PRIMARY' else 'Original SCALE auxiliary only; not primary and not a repeated-outlier guarantee')
    selection = {'created_utc': now, 'status': 'FROZEN_NEXT5_FINAL_ROLES', 'roles': roles,
        'new_NEXT5_roles': [], 'selected_primary': 'FROZEN_PRIMARY',
        'new_adopted_primary_estimated_increment': 0.0,
        'increment_interpretation': 'No NEXT5 improvement is adopted or claimed; not a new hidden-test estimate',
        'best_observed_ResearchPool_candidate': full[0]['candidate'],
        'best_observed_ResearchPool_mean': full[0]['mean'],
        'best_observed_ResearchPool_delta': full[0]['mean_delta'],
        'best_observed_candidate_is_selected': False, 'new_generation': False,
        'B2_role': 'Historical exploratory near-miss only, never a deployment champion',
        'finalist_manifest_sha256': digest(ROOT / 'FINALIST_MANIFEST_NEXT5.json'),
        'frozen_reference_receipt_sha256': digest(refs_path),
        'shadow_status': 'CLOSED', 'new_seal_rows': 0, 'new_reduced_uses': 0,
        'new_public_submissions': 0, 'new_cloud_submissions': 0,
        'fresh_raw_reproduction': 'NOT_REQUIRED_NO_NEW_NEXT5_ROLE',
        'engineering_audit_and_final_delivery_still_required': True}
    atomic_json(ROOT / 'DEV_CANDIDATES_NEXT5.json', selection, immutable=True)
    with (ROOT / 'SHADOW_RESULTS_NEXT5.csv').open('x', encoding='utf-8', newline='') as stream:
        fields = ['status', 'finalists', 'evaluations', 'candidate', 'TS_AUC', 'delta_vs_BL80', 'reason']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({'status': 'NOT_EVALUATED_CLOSED', 'finalists': 0, 'evaluations': 0,
            'reason': manifest['reason']})
    guard.check_state()
    progress('Finalist set frozen without opening Shadow', 'No eligible candidates; retain original BL80 primary and original D2 SCALE auxiliary')
    print('NEXT5 FINALISTS FROZEN: 0; SHADOW CLOSED; ORIGINAL PRIMARY RETAINED', flush=True)


if __name__ == '__main__':
    main()
