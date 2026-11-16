"""First complete crossed-cohort review, without changing any scientific recipe."""
from pathlib import Path
import csv
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.next7.common import BASE, save, read_json, csv_write, digest, utc_now, activity, note


def rows(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def main():
    previous_path = BASE / 'bottlenecks/NEXT7_BOTTLENECK_CYCLE_004.json'
    previous = read_json(previous_path)
    items = {r['id']: r for r in previous['items']}
    completed = sorted((BASE / 'conditional_statistics').glob('*/*/DONE.json'))
    assert completed
    selected = []
    with activity('bottleneck cycle005: conditional H risk versus retained early crossing', 'BOTTLENECK_FORENSIC', 'F4_NULL_GROWTH_CORRECTION'):
        for path in completed:
            assert read_json(path)['status'] == 'COMPLETE'
            selected += [r for r in rows(path.parent / 'SUMMARY.csv') if r['horizon'] == '16384']
        target = {r['metric']: r for r in selected if r['case'] == 'N0' and r['H_length'] == '1000'}
        q, terminal, early = (target[n] for n in ('G2_QUALIFIED_EVER_LATCH',
            'FIXED_HORIZON_MAX_EXCEEDS_TERMINAL_CUT', 'EARLIER_CROSSING_RETAINED_ONLY'))
        items['B4'].update(status='FIRST_COMPLETE_COHORT_TERMINAL_EXCEEDANCE_DOMINATES_EARLY_RETENTION',
            evidence=f"N0,H1000,16384: qualified rate={q['mean_crossed_rate']}, terminal-cut exceedance={terminal['mean_crossed_rate']}, earlier-retained-only={early['mean_crossed_rate']}. Partition exact in every cell and matched resample. This cohort's long-null error is mostly not attributable to early-boundary retention alone.",
            next_test='Complete all12 locked cohorts and exact empirical-support partitions; no horizon or cutoff changes.',
            confidence='Completed512x512 grid, approximate same-seed-pair bootstrap; dependent crossed cells, not262144 independent trials.')
        items['B5'].update(status='LARGE_CONDITIONAL_H_HETEROGENEITY_OBSERVED_IN_COMPLETE_COHORT',
            evidence=f"N0,H1000,16384: conditional-H median={q['conditional_H_rate_quantile_0.5']};90th percentile={q['conditional_H_rate_quantile_0.9']}; fraction of H with risk>.5={q['fraction_H_conditional_rate_above_0.5']}. H main-effect share of complete-grid cell variation={q['H_main_effect_fraction_total_variance']}, a descriptive decomposition, not causal attribution.",
            next_test='CompareH2048/H5000 and AR/ARCH/heavy-tail cohorts. Exact finite empirical-bootstrap support enumerator queued to locate unreachable future evidence.',
            confidence='No conditional-H5% guarantee, no current result extrapolation to incomplete cohorts.')
        items['B2'].update(next_test='Same fitted-model classifier calibration and all23 stress cases queued. Uncertainty implementation frozen: common observed support, fixed-fold means within seed draws, missing late recovery excluded.')
        items['B6'].update(status='PORTABLE_RESEARCH_BUNDLES_PASS_FULL_TRAINING_GRADIENT_FORENSIC_QUEUED',
            evidence='Five research-only fold bundles exported without fitting. Three tests passed: portable streaming with original data/artifact reads denied, cross-fold head mix blocked, raw/qualified mode mismatch blocked. Full G3 training-gradient forensic uses all original pairs at frozen coefficients; no optimization or new pairs.',
            next_test='Complete aggregate sampling-error diagnostic and execute fixed quiet scalar runtime workload during admitted final audit.')
        items['B7'].update(next_test='Owned real crash/resume test, source authority recheck, final audit admission after36000 productive seconds and research closure. No new model search.')
        heartbeat = read_json(BASE / 'runtime/HEARTBEAT.json')
        result = {'created_utc': utc_now(), 'cycle': 5, 'status': 'RESEARCH_CONTINUES',
            'wall_seconds': heartbeat['wall_seconds'], 'productive_seconds': heartbeat['productive_seconds'],
            'priority': ['B5', 'B4', 'B2', 'B6', 'B3', 'B1', 'B7'], 'items': list(items.values()),
            'current_primary': 'FROZEN_BL80', 'learned_candidate_result': 'NO_PROMOTION',
            'research_duration_requirement_not_yet_met': True,
            'prior_cycle_sha256': digest(previous_path), 'complete_cohort_long_null_results': selected,
            'evidence_sha256': {p.relative_to(BASE).as_posix(): digest(p) for p in completed},
            'additional_evidence_sha256': {n: digest(BASE / n) for n in ('RESEARCH_BUNDLES_EXPORTED.json',
                'SUPPORT_PARTITION_IMPLEMENTATION_LOCK.json', 'CLASSIFIER_UNCERTAINTY_IMPLEMENTATION_LOCK.json',
                'G3_FULL_GRADIENT_FORENSIC_LOCK.json', 'LATENCY_WORKLOAD_LOCK_NEXT7.json')}}
        save('bottlenecks/NEXT7_BOTTLENECK_CYCLE_005.json', result)
        save('runtime/BOTTLENECK_BOARD_CURRENT.json', result, False)
        history = rows(BASE / 'BOTTLENECK_HISTORY_NEXT7.csv')
        history += [{'cycle': 5, 'created_utc': result['created_utc'], 'id': r['id'], 'status': r['status'],
            'severity': r['severity'], 'evidence': r['evidence'], 'next_test': r['next_test']} for r in items.values()]
        csv_write('BOTTLENECK_HISTORY_NEXT7.csv', history, immutable=False)
    note('Bottleneck cycle005 complete', 'First full conditional cohort identifies H-dependent threshold transfer as larger than early-memory retention', 'F4_NULL_GROWTH_CORRECTION')


if __name__ == '__main__':
    main()
