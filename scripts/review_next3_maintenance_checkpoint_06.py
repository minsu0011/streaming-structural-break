"""Register bounded post-closure diagnostics and review completed engineering."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, atomic_json, digest, progress, read_json, utc_now
from src.next3.bottlenecks import record_board
from src.next3.maintenance import register_job


def main():
    for name, protocol, values in (
        ('D006_LEGACY_STATE_REFINEMENT', 'NEXT3_LEGACY_STATE_REFINEMENT_PROTOCOL.md', {
            'HYPOTHESIS': 'D005 legacy-branch persistence can be localized to seven existing recurrent substates using the same original paired trajectories.',
            'BOTTLENECK': 'B6',
            'CHANGE': 'Evaluation-only matched substate transfers; no model, observation, horizon or threshold changes.',
            'EXPECTED_EFFECT': 'Distinguish transient legacy filtering from persistent EWMA/CUSUM/running-mean state in both intervention directions.',
            'RISKS': 'Unreachable hybrid states, nonadditivity and adaptive diagnostic choice preclude unique real-DEV causal attribution.',
            'PRIMARY_METRIC': 'The same probability and frozen-threshold intervention contrasts as D005; no null TS-AUC.',
            'SECONDARY_DIAGNOSTIC': 'Every original prediction, clone and complete fine-to-coarse transfer exact; immutable arrays and constant state.',
            'STOP_RULE': 'Stop on any invariant mismatch; no further scientific fits or deployment change.',
            'bootstrap_seed': 2026091071, 'bootstrap_replicates': 2000,
        }),
        ('D007_PAIR_MATRIX_BOOTSTRAP_AUDIT', 'NEXT3_PAIR_MATRIX_AUDIT_PROTOCOL.md', {
            'HYPOTHESIS': 'An independent integer pair-matrix formulation reproduces all frozen original bootstrap scores and conditional conclusions.',
            'BOTTLENECK': 'B4',
            'CHANGE': 'Numerical audit of existing predictions using explicit pair enumeration instead of the rank-sum kernel.',
            'EXPECTED_EFFECT': 'Validate tie handling, whole-series duplicate weights, fold aggregation and conditional uncertainty reporting.',
            'RISKS': 'Reproducing the same conditional statistic does not remove adaptive selection bias; pair matrices require bounded memory.',
            'PRIMARY_METRIC': 'Absolute discrepancy of existing observed/bootstrap scores, maximum allowed 1e-14.',
            'SECONDARY_DIAGNOSTIC': 'Reconstructed fixed individual intervals and joint conditional envelope, zero new fits and unchanged original seeds.',
            'STOP_RULE': 'Any discrepancy stops the audit and preserves the failure; no metric tuning or scientific promotion.',
            'pool_policy_sha256': digest(BASE/'selection_uncertainty/POLICY.json'),
            'existing_results_sha256': digest(BASE/'selection_uncertainty/RESULTS.json'),
        }),
    ):
        atomic_json(BASE/'hypotheses'/f'{name}.json', {
            'created_utc': utc_now(), **values, 'protocol_sha256': digest(ROOT/'docs'/protocol),
            'new_fits': 0, 'new_scientific_candidates': 0}, immutable=True)
    board = read_json(BASE/'bottlenecks/BOTTLENECK_BOARD.json')['bottlenecks']
    for item in board:
        if item['id'] == 'B8':
            item.update(status='PARTIALLY_RESOLVED',
                next_test='All three engineering families closed; lean runtime passes fixed parity/cost gates. Final raw-data reproduction and actual archived-source replay remain mandatory.')
            for name in ('engineering/lean_shared/ENGINEERING_GATE_DECISION.json',
                         'lean_public_api_all_dev/RESULTS.json', 'portable_runtime_golden/RESULTS.json'):
                path = BASE/name
                item['evidence'].append({'path': path.relative_to(ROOT).as_posix(), 'sha256': digest(path)})
        elif item['id'] == 'B6':
            item.update(status='ACTIVE', next_test='Summarize all D005 windows/directions and perform registered seven-block legacy refinement on the identical O1 cohort; no candidate remedy.')
            path = BASE/'outlier_state_transfer/run_v1/RESULTS.json'
            item['evidence'].append({'path': path.relative_to(ROOT).as_posix(), 'sha256': digest(path)})
        elif item['id'] == 'B4':
            item.update(status='ACTIVE', next_test='D007 independent pair-matrix audit of every original fixed bootstrap replicate; preserve adaptive-DEV limitation and no new scientific selection.')
    record_board(board, 'E003 eligible with 26.35% lower cold RSS and unchanged 10M cost/parity; 685 tests and raw-free portability fixture pass. B8 engineering menu closed; B6 state characterization and B4 numerical uncertainty audit are active.')
    for job_id, argv, description in (
        ('M601_PROGRESS_EXPORT', ['scripts/export_next3_progress.py'], 'Refresh completed-only reporting after lean gates, 685 passing tests and third planned supervisor replacement'),
        ('M602_LINEAGE_CHECKPOINT', ['scripts/audit_next3_lineage.py', '--label', 'checkpoint_06'], 'Recheck protected old bytes and nested/OOF provenance after new runtime and maintenance changes'),
    ):
        register_job({'id': job_id, 'kind': 'maintenance', 'argv': argv,
                      'description': description, 'retry_safe': False})
    progress('NEXT3 checkpoint 06', 'B8 engineering family closed; registered B6 legacy refinement and B4 independent pair-matrix audit before execution')
    print('NEXT3 CHECKPOINT 06 REGISTERED', flush=True)


if __name__ == '__main__':
    main()
