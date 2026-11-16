"""Prospective details before any NEXT9 break label or screen score inspection."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.next9
from src.next9.common import BASE, atomic, digest, read_json, utc_now, activity
from src.next9.orientation_contract import PAIR_MENU


def main():
    assert not (BASE / 'LABEL_ACCESS_GATE_NEXT9.json').exists()
    with activity('Freeze finite orientation pair and honest correction qualification details', 'PROTOCOL', 'B1_COMMON_CURRENT_RANGE'):
        policy = dict(status='FROZEN', timestamp=utc_now(),
            protocol_sha256=digest(BASE / 'NEXT9_PROTOCOL_LOCK.json'),
            new_screen_quality_inspected_by_controller=False, new_break_labels_read=False,
            pairs=[dict(family=p[0], first=p[1], second=p[2]) for p in PAIR_MENU],
            pair_count_per_feature=8, pair_selection='No additional pair menu; no ranking by outer labels',
            common_range_quantile_weighting='Empirical observation quantiles within TRAIN state, q10/q90. Original-series group support and group-weighted model fit remain separate.',
            RUN_sparse_mapping='Unsupported child states map to unique codes10,11,12 for S1 DOWN,FLAT,UP; supported children retain0..8; UNKNOWN9. Pair-specific run evidence requires its original two children supported. Collapsed parents may be evaluated only by existing S1 pair, never counted as new pair.',
            S3_role='Prespecified weak/strong support and effect diagnostics within S1 model; no fifth state-family or extra learner',
            state_only='Movement and reversal main effects; comparison matched to O5. Additional matched state-only for O3/O4 uses their own fixed state vocabulary.',
            O6_matrix='Exactly O5 numeric inputs including fixed nonlinear F and explicit interactions, same rows and weights',
            effect_diagnostic='Heldout common-range state-specific signed TS-AUC(F) minus0.5; whole-original-group paired difference test between states. TRAIN-fitted interaction coefficients reported separately. No heldout coefficient refit for deployed rule.',
            multiplicity='Holm over all8pairs times all frozen candidate features within each TRAIN procedure and mode. Missing or unsupported tests receive p1; family never shrinks after outcomes.',
            provisional_rule='Same pair and opposite pattern in at least2/3 internal heldout panels, sufficient common range, Holm-adjusted effect difference, positive paired O versus O1 and matched STATE_ONLY logloss and TS-AUC. Tie by frozen movement rank then pair menu order; maximum6 features. Frozen TRAIN coefficient applies only inside TRAIN common range and observed supported states.',
            M7_qualification=dict(
                diagnostic='Single fixed correction diagnostic p=.95*matched_BL80+.05*inner_excluded_M5. No alternative weight and no grid.',
                gate='Original-group paired TS-AUC95% lowerCI>0 and at least2/3 positive original inner folds; evaluated only from inner excluded procedure predictions.',
                reason='Fitting a meta correction and evaluating it against the labels used for that meta fit would not be heldout. The fixed diagnostic needs no meta fit and does not leak heldout labels into its coefficient.',
                accepted='Fit the already specified C=.1 offset ridge-logistic correction using all honest inner OOF predictions and TRAIN labels. Its learned quality is assessed only on original outer holdout after global freeze.',
                rejected='Exact BL80. Record learned correction unused and zero correction training fits.',
                claims='Diagnostic CI is for the fixed5percent correction only. Never describe it as an unbiased inner estimate of the subsequently learned M7.',
                no_extra_scientific_model='Qualification diagnostic only; final model menu unchanged.'),
            early='Every rule selection and learned coefficient uses training and internal validation observations t<=199. Late heldout used only after full early freeze.',
            limits='All final certificates require original outer4of5, time/H replication and early-late. Internal provisional selection is not a certificate.')
        atomic(BASE / 'ORIENTATION_DETAILS_PROTOCOL_NEXT9.json', policy)
        atomic(BASE / 'ORIENTATION_CONTRACT_IMPLEMENTATION_LOCK_NEXT9.json', dict(status='FROZEN', timestamp=utc_now(),
            files_sha256={p:digest(ROOT / p) for p in ['src/next9/orientation_contract.py', 'scripts/next9/freeze_next9_orientation_details.py', 'scripts/next9/tests/test_orientation_contract.py']},
            policy_sha256=digest(BASE / 'ORIENTATION_DETAILS_PROTOCOL_NEXT9.json'),
            tests_sha256=digest(BASE / 'tests/ORIENTATION_CONTRACT_TESTS.xml'), test_count=7, tests_status='PASS'))
        heart = read_json(BASE / 'runtime/HEARTBEAT_NEXT9.json')
        atomic(BASE / 'checkpoints/NEXT9_CONTROLLER_CHECKPOINT_001.json', dict(timestamp=utc_now(), **{k:heart[k] for k in ['productive_seconds','phase','active_bottleneck','target_free_candidates','certified_variable_count','cross_edge_count','inner_best','outer_released','failed_tasks']},
            resource_status={k:heart[k] for k in ['RAM_percent','disk_free_bytes','queue_capacity']},
            next_task='Complete all153 screen and early-only selection, then freeze20 candidate procedures. Implement scoped label gate and matched orientation/transition estimators without accessing new labels.',
            source_hash=digest(ROOT / 'scripts/next9/freeze_next9_orientation_details.py'),
            review_due_UTC='2026-09-14T12:44:05+00:00', labels_read=False,
            software_repair='One float32 epsilon-floor representation error repaired before screen; failed evidence preserved. No queued task failures.'))


if __name__ == '__main__':
    main()
