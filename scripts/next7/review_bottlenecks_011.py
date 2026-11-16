"""Review the complete AR crossed cohorts and explicit unmet design requirements."""
from pathlib import Path
import csv
import datetime as dt
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.next7.common import BASE, read_json, save, csv_write, digest, utc_now, activity, note


def main():
    previous = BASE/'bottlenecks/NEXT7_BOTTLENECK_CYCLE_010.json'
    prior = read_json(previous)
    elapsed = (dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(prior['created_utc'])).total_seconds()
    assert elapsed >= 45*60
    items = {r['id']: r for r in prior['items']}
    with activity('Bottleneck cycle011: complete AR H-length contrasts and design requirements',
                  'BOTTLENECK_FORENSIC', 'F4_NULL_GROWTH_CORRECTION'):
        items['B1'].update(next_test='Retain completed V1-V4 classifier gains and weak-contraction limits. Finish the remaining fixed mechanisms without refitting or changing cutoffs.')
        items['B2'].update(status='SHOCK_ROLE_GATE_DESIGN_LIMITATION_EXPLICITLY_REGISTERED',
            evidence='The user requested a large shock false-latch reduction without a numeric amount. The initial agent protocol quantified this as an absolute0.10 reduction but did not fully specify reference, endpoint and horizon. Several frozen baseline rates are below0.10, so that reduction is empirically impossible for those endpoints. A registered24-row feasibility audit preserves all state/classifier, reference and horizon rows, including missing M3 paired-control fields. All15 candidates independently fail the user worst-fold non-worsening condition.',
            next_test='Finish the original classifier shock/control cohorts and the registered gate feasibility audit. Preserve the gate and missing fields; do not select a favorable endpoint after seeing results.')
        items['B3'].update(status='STRICT_CORE_RAW_BLOCK_DISJOINTNESS_REQUIREMENT_NOT_MET',
            evidence='The new scale channel uses disjoint current16-point raw blocks. The inherited core lag4 features reuse up to four raw observations across adjacent feature blocks, so strict nonoverlap of all raw input footprints is NOT_MET. Finite shock influence and preservation of the old baseline do not waive this user requirement. The handoff must state this unmet design requirement and NO_PROMOTION, rather than claiming a production-ready state network.',
            next_test='Preserve this explicit limitation in the closure, final audit and45-question handoff. A strict-disjoint control belongs to a future separately registered model generation, not a silent change to frozen NEXT7 inference.')
        items['B4'].update(status='AR_LONG_FALSE_MEMORY_NONMONOTONE_WITH_H_LENGTH',
            evidence='N1 crossed G2 long false memory atH1000/2048/5000 is.3887519836425781/.2753181457519531/.3211517333984375. Matched H5000-H2048 delta=.045833587646484375, pointwise interval[.012847423553466797,.08157224655151361]. H5000-H1000 delta=-.06760025024414062, interval[-.10667781829833985,-.027096939086914077]. These complete fixed contrasts reject a simple monotone H-length remedy within this experiment; they do not select a new operating H.',
            next_test='Complete all six N2/N3 crossed cohorts and preserve all original H-length contrasts without choosing a new H from the result.')
        items['B5'].update(status='LARGER_AR_H_REDUCES_SUPPORT_EXCEEDANCE_BUT_INCREASES_WITHIN_SUPPORT_TAIL',
            evidence='AtN1,H5000,16384 the exact G2 partition is beyond empirical support.109100341796875, within support above terminal cut.1483612060546875, earlier retained only.063690185546875. VersusH2048, beyond-support probability drops.101104736328125, but within-support probability rises.1483612060546875; retained-only delta=-.001422882080078125 has an interval crossing zero. H5000 rank-tail log spread mismatch correlates with conditional risk r=.5376768377309752, interval[.4656915494554573,.5977382388352055], posthoc and noncausal. Exact finite-multinomial moment auditing remains registered for all12 cohorts.',
            next_test='Complete every background and the108-row arithmetic audit. Separate support limitations, transported normalization and retained history without claiming these descriptive associations identify a causal repair.')
        items['B6'].update(next_test='All five exact full-pair objective diagnostics are complete. Retain the sampled/exact gradient discrepancy and held-out failure; measure runtime only during the quiet final audit.')
        items['B7'].update(status='THREE_OWNED_WORKERS_RUNNING_ENGINEERING_FAILURES_RESOLVED',
            evidence='A resource-only scheduler maintenance increased capacity from two to three, with six ownership tests passing and the two live scientific jobs adopted without duplicate execution. The temporary maintenance stop was removed. Original A1/A2 ARCH-persistence classifier cohorts are complete: A2 G3 correct first alarm at1024 improves.16171875 versus M0, interval[.14140625,.18359375], while overall RP promotion still fails. A report preflight incorrectly assumed six H regimes; the original defines five. The corrected127-binding preflight passes, and the preserved eight-failure registry has zero unresolved items. Queue contains115 tasks; final audit and delivery remain gated.',
            next_test='Finish all queued research and actual productive time, then full audit and quiet scalar benchmarks. Verify both portable replay entry points from the final ZIP and copy exactly20 desktop files.')
        paths = ['conditional_statistics/N1/H_5000/DONE.json', 'support_partition/N1/H_5000/DONE.json',
            'crossed_H_length_comparison/N1/DONE.json', 'normalizer_transport/correlations/N1/H_5000/DONE.json',
            'COMPLETE_DIAGNOSTICS_GRADIENTS_NEXT7.json', 'classifier_statistics/A1/DONE.json',
            'classifier_statistics/A2/DONE.json', 'CROSSED_MOMENT_AUDIT_LOCK_NEXT7.json',
            'ROLE_GATE_FEASIBILITY_LOCK_NEXT7.json', 'FRESH_BLOCK_REQUIREMENT_AUDIT_NEXT7.json',
            'audit/THREE_WORKER_MONITOR_RESTART_001.json', 'audit/CORE_HANDOFF_BINDING_PREFLIGHT_002.json',
            'audit/failure_registry/research_002/DONE.json']
        h = read_json(BASE/'runtime/HEARTBEAT.json')
        result = {'created_utc': utc_now(), 'cycle': 11, 'status': 'RESEARCH_CONTINUES',
            'wall_seconds': h['wall_seconds'], 'productive_seconds': h['productive_seconds'],
            'priority': ['B5','B4','B3','B2','B1','B6','B7'], 'items': list(items.values()),
            'current_primary': 'FROZEN_BL80', 'learned_candidate_result': 'NO_PROMOTION',
            'research_duration_requirement_not_yet_met': not h['research_minimum_met'],
            'prior_cycle_sha256': digest(previous), 'evidence_sha256': {name:digest(BASE/name) for name in paths},
            'no_new_candidate_fit_or_threshold_or_random_draw': True}
        save('bottlenecks/NEXT7_BOTTLENECK_CYCLE_011.json', result)
        save('runtime/BOTTLENECK_BOARD_CURRENT.json', result, False)
        with (BASE/'BOTTLENECK_HISTORY_NEXT7.csv').open(newline='', encoding='utf-8-sig') as handle:
            history = list(csv.DictReader(handle))
        history += [{'cycle':11, 'created_utc':result['created_utc'], 'id':r['id'], 'status':r['status'],
            'severity':r['severity'], 'evidence':r['evidence'], 'next_test':r['next_test']} for r in items.values()]
        csv_write('BOTTLENECK_HISTORY_NEXT7.csv', history, immutable=False)
    note('Bottleneck cycle011 complete', 'AR H-length nonmonotonicity and explicit unmet design requirements; no promotion', 'F4_NULL_GROWTH_CORRECTION')


if __name__ == '__main__':
    main()
