"""Freeze the two unchanged scientific roles without claiming the final audit."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, atomic_json, digest, progress, read_json, utc_now
from src.next.guard import SealGuard


def main():
    guard = SealGuard(ROOT); guard.check_state()
    closure_path = BASE/'bottlenecks/FINITE_MENU_CLOSURE_REVIEW.json'; closure = read_json(closure_path)
    assert closure['status'] == 'FINITE_SCIENTIFIC_MENU_CLOSED_MAINTENANCE_CONTINUES'
    assert closure['performance_promotions'] == closure['robustness_promotions'] == 0
    gate_path = BASE/'selection/NECESSARY_GATES.json'; gates = read_json(gate_path)
    assert len(gates['decisions']) == 13
    for decision in gates['decisions']:
        assert not decision['promotion']
        assert decision['performance_necessary_gates'] in ('NECESSARY_GATE_FAIL', 'SCREEN_FAIL_STOP')
        assert decision['robustness_necessary_gates'] in ('NECESSARY_GATE_FAIL', 'SCREEN_FAIL_STOP')
    references_path = BASE/'preflight/FROZEN_REFERENCES.json'; references = read_json(references_path)
    roles = {}
    for name in ('FROZEN_DISTILLED', 'FROZEN_PERFORMANCE'):
        reference = references['roles'][name]
        model = ROOT/reference['model_directory']/'model.sbnext2'
        assert digest(model) == reference['canonical_model_sha256']
        roles[name] = {'experiment_id': reference['experiment_id'],
            'reference_model_directory': reference['model_directory'],
            'canonical_model_sha256': reference['canonical_model_sha256'],
            'model_bytes': model.stat().st_size,
            'original_five_fold_scores': reference['full']['fold_scores'],
            'original_mean': reference['full']['mean'], 'original_median': reference['full']['median'],
            'original_worst_fold': reference['full']['worst_fold'],
            'quality_change_since_session_start': 0.0, 'new_scientific_fit_required_for_selection': False}
    lean = read_json(ROOT/'configs/next3/LEAN_RUNTIME_SELECTION.json')
    assert lean['status'] == 'PROVISIONAL_ENGINEERING_ELIGIBLE'
    out = ROOT/'configs/next3/SCIENTIFIC_ROLE_FREEZE.json'
    record = {'status': 'PRE_AUDIT_SCIENTIFIC_ROLES_FROZEN', 'created_utc': utc_now(),
        'roles': roles, 'new_performance_candidate': None, 'new_robustness_candidate': None,
        'scientific_role_count': 2, 'generation_gate_passed': False,
        'new_scientific_search_authorized': False,
        'engineering_variant_provisionally_selected': lean['variant'],
        'engineering_selection_sha256': digest(ROOT/'configs/next3/LEAN_RUNTIME_SELECTION.json'),
        'engineering_is_the_same_scientific_performance_role': True,
        'engineering_final_acceptance_pending': ['fresh_pinned_raw_training_reproduction', 'final_audit', 'actual_archive_verification'],
        'fallback_rule': 'If final engineering verification fails, preserve the scientific lock and document a return to a previously verified runtime; never substitute a rejected scientific candidate.',
        'reference_declaration_sha256': digest(references_path), 'finite_menu_closure_sha256': digest(closure_path),
        'necessary_gate_receipt_sha256': digest(gate_path),
        'all_seals_unopened': True, 'reduced_new_uses': 0, 'public_new_submissions': 0,
        'final_audit_completed': False, 'scope': 'Scientific role freeze only; runtime delivery acceptance remains a separately verified final-audit action.'}
    atomic_json(out, record, immutable=True)
    progress('NEXT3 scientific roles frozen', 'Two original scientific roles only; engineering acceptance awaits final raw reproduction and archive proof')
    print('NEXT3 SCIENTIFIC ROLE FREEZE', len(roles), 'NO NEW SCIENTIFIC PROMOTION', flush=True)


if __name__ == '__main__': main()
