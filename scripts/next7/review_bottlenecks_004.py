"""Evidence review after target alignment, exact gradients and runtime integration."""
from pathlib import Path
import csv
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.next7.common import BASE, save, read_json, csv_write, digest, utc_now, activity, note


def rows(name):
    with (BASE / name).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def main():
    with activity('bottleneck cycle004: rank accounting and classifier calibration separation', 'BOTTLENECK_FORENSIC', 'F5_CAUSAL_NESTED_INTEGRATION'):
        previous = read_json(BASE / 'bottlenecks/NEXT7_BOTTLENECK_CYCLE_003.json')
        items = {r['id']: r for r in previous['items']}
        alignment = rows('TARGET_ALIGNMENT_RESULTS_G2.csv')
        pairs = rows('forensic_g2_g3/MEAN_MEMORY_PAIR_DECOMPOSITION.csv')
        exact = rows('G3_EXACT_VS_SAMPLED_PAIR_DIAGNOSTICS.csv')
        cosine = [float(r['gradient_cosine']) for r in exact if r.get('beta') == 'G3_FITTED' and r.get('gradient_cosine')]
        items['B1'].update(next_test='Queued model-specific calibrated I4/G2/G3 classifier scale stress; preserve fixed V3/V4 and0.9M0 controls.')
        items['B2'].update(status='CLASSIFIER_VERSUS_STATE_ALARMS_EXPLICITLY_SEPARATED',
            evidence='Classifier stress protocol frozen for five existing NEXT7 definitions, original five fitted folds, shared null paths. M0/M3 metric replay parity passed20 checks; own-calibrated0.9M0 must have identical alarm paths.',
            next_test='Finish1024 null calibration paths before all23 classifier stress cases; no cuts chosen from evaluation.')
        items['B3'].update(status='EXACT_FALSE_MEMORY_BENEFIT_AND_MISSED_CHANGE_COST_QUANTIFIED',
            evidence='RP no-break raw latch.1028908921 becomes qualified.0774013056; first latch after observed break.0942507069 becomes.0760289036. For G2_EVENT versus raw M_EVENT, removing negative-only memory contributes+.0030190744 TS-AUC while removing positive-only memory contributes-.0027353016; total gain only+.0003284032 and final delta versus M0 remains negative.',
            next_test='Keep causal memory mechanics separate from discriminative ranking; cross-reference classifier stress and H-reference uncertainty.',
            confidence='Exact same-time positive-negative rank accounting across all frozen folds; posthoc algebra, not causal effect identification.')
        items['B4'].update(next_test='Crossed null grid already running: quantify earlier retained boundary crossings versus fixed-horizon maximum exceedance.',
            confidence='Existing512-path intervals complete; ongoing512-by512 cross reuses seeds and has no independent-cell or anytime guarantee.')
        items['B5'].update(status='CONDITIONAL_CROSSED_GRID_RUNNING_AFTER_EXACT_DIAGONAL_GATE',
            evidence='2048 raw generator diagonal paths and6144 state/H diagonals match original generator and H-transport results bitwise. Two owned workers run the fixed grid with complete per-H row receipts.',
            next_test='After complete512-reference cohorts, report conditional-H false-latch distributions and crossed seed dependence; do not recalibrate.')
        items['B6'].update(status='INTEGRATED_RUNTIME_PARITY_PASS_SAMPLING_LIMIT_QUANTIFIED',
            evidence=f'19 integrated tests passed for static, logistic and tree heads. G2/G3 change BL80 rankings substantially (Pearson below.995 on every fold) yet lose overall score. Prespecified-time G3 sampled/exact gradient cosine median={float(np.median(cosine)):.6f}; local sampling error is visible, but does not establish the cause of held-out loss.',
            next_test='Training-only aggregate sampling-error forensic if warranted, plus quiet final scalar latency and reproduction. No new learner, cap, seed or penalty.',
            confidence='Frozen five-fold outcomes and exact selected training strata; no universal surrogate or sampling impossibility claim.')
        items['B7'].update(status='BUDGET_CLOSED_MONITOR_CAPACITY_AND_PHASE_GATES_TESTED',
            evidence='15 full scientific candidates remain unchanged. Queue pre-reserves every active worker before dispatch; seven maintenance tests passed. Formal audit requires36000 productive seconds and concrete research closure. Work units are not counted as new scientific candidates.',
            next_test='Continue productive failure analysis; preserve original source artifacts and final audit separation.')
        h = read_json(BASE / 'runtime/HEARTBEAT.json')
        result = {'created_utc': utc_now(), 'cycle': 4, 'status': 'RESEARCH_CONTINUES',
            'wall_seconds': h['wall_seconds'], 'productive_seconds': h['productive_seconds'],
            'priority': ['B5', 'B4', 'B3', 'B2', 'B6', 'B1', 'B7'], 'items': list(items.values()),
            'current_primary': 'FROZEN_BL80', 'learned_candidate_result': 'NO_PROMOTION',
            'research_duration_requirement_not_yet_met': True,
            'prior_cycle_sha256': digest(BASE / 'bottlenecks/NEXT7_BOTTLENECK_CYCLE_003.json'),
            'target_alignment': alignment,
            'evidence_sha256': {n: digest(BASE / n) for n in ('forensic_g2_g3/DONE.json',
                'G3_PAIR_DIAGNOSTIC_DONE.json', 'CLASSIFIER_STRESS_PROTOCOL_LOCK.json',
                'CLASSIFIER_STRESS_PREFLIGHT_GATE.json', 'runtime/SCHEDULER_UPGRADE_002_DONE.json')}}
        save('bottlenecks/NEXT7_BOTTLENECK_CYCLE_004.json', result)
        save('runtime/BOTTLENECK_BOARD_CURRENT.json', result, False)
        history = rows('BOTTLENECK_HISTORY_NEXT7.csv')
        history += [{'cycle': 4, 'created_utc': result['created_utc'], 'id': r['id'], 'status': r['status'],
            'severity': r['severity'], 'evidence': r['evidence'], 'next_test': r['next_test']} for r in items.values()]
        csv_write('BOTTLENECK_HISTORY_NEXT7.csv', history, immutable=False)
        save('runtime/CURRENT_RESEARCH_STATUS.json', {'best': 'I4_EVENT_SCALE_TREE',
            'best_mean_delta': -.0003624879784099866, 'bottleneck': 'B5_CONDITIONAL_H_UNCERTAINTY',
            'next_task': 'Conditional-grid and classifier stress completion, crossed uncertainty, quiet runtime, reproducibility',
            'screen_count': 7, 'full_count': 15, 'created_utc': utc_now()}, False)
    note('Bottleneck cycle004 complete', 'Memory filtering helps some rank pairs but removes useful changes; full classifier stress queued', 'F4_NULL_GROWTH_CORRECTION')


if __name__ == '__main__':
    main()
