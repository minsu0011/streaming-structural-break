"""A crossed fixed-innovation diagnostic, not a new prediction candidate."""
from pathlib import Path
import csv
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.next7.common import BASE, save, read_json, digest, csv_write, utc_now, activity, note


def main():
    contract = {'status': 'FROZEN', 'created_utc': utc_now(), 'id': 'CONDITIONAL_NULL_REFERENCE_FORENSIC',
        'purpose': 'Separate H-reference uncertainty, fixed-horizon bootstrap undercoverage, and retention after an early boundary crossing.',
        'not_a_candidate': True, 'new_model_fits': 0, 'new_RP_scores': 0,
        'backgrounds': ['N0', 'N1', 'N2', 'N3'], 'partition': 33, 'reference_indices': [0, 511],
        'innovation_indices': [0, 511], 'H_lengths': [1000, 2048, 5000], 'online_horizon': 16384,
        'H_source': 'Existing canonical null generator with5000 H observations. Use its latest H_length observations. Keep exact unrounded terminal DGP y and squared innovation for the synthetic continuation only.',
        'innovations': 'Exactly the existing512 pre/post child noise streams per background, using the original seed/partition/scenario ordinal/index. No new random seed or seed replacement.',
        'crossing': 'For each H_i, replay each existing innovation stream j from the exact terminal DGP state of H_i. The detector receives only H_float32 and each current float32 observation.',
        'causality': 'The synthetic generator uses its known null parameters; those parameters and latent DGP state never enter detector features or thresholds.',
        'diagonal_test': 'i=j raw H/O must match the original generator and fixed H-transport cases exactly before the crossed grid is admitted.',
        'precision_test': 'Fused execution must match every reference G1/G2 output bitwise on fixed prefix/recovery/unsupported-H tests before dispatch.',
        'metrics': ['raw/core/qualified ever-latch at1024/4096/16384', 'first latch time',
            'maximum raw confirmation ratio through each of the nine fixed G2 block grids',
            'maximum ratio divided by the contemporaneous routed H cut',
            'fixed-horizon maximum exceeding that horizon cut versus any earlier retained boundary crossing'],
        'interpretation': 'A512-by512 crossed matrix reuses512 histories and512 innovation streams. It is not262144 independent seeds or an untouched validation set.',
        'uncertainty': 'Report the per-H distribution and crossed dependence explicitly; use existing paired whole-seed bootstrap counts on both axes for population summaries. No new draws to change decisions.',
        'forbidden': ['Any recalibration of G1/G2 from these outcomes', 'Selection of a new threshold, horizon, weight, learner or seed',
            'Promotion based on these diagnostic null results', 'Treating unknown DGP parameters as model inputs'],
        'execution': 'Immutable source/test gate, at most two owned queued workers, row-level resumable artifacts, CPU/output monitoring and one safe retry.',
        'H_transport_results_sha256': digest(BASE / 'H_LENGTH_CALIBRATION_NEXT7.csv'),
        'G2_protocol_sha256': digest(BASE / 'G2_PROTOCOL_LOCK.json')}
    save('CONDITIONAL_NULL_FORENSIC_LOCK.json', contract)
    with activity('bottleneck cycle003: learned families closed, conditional null uncertainty', 'BOTTLENECK_FORENSIC', 'F4_NULL_GROWTH_CORRECTION'):
        previous = read_json(BASE / 'bottlenecks/NEXT7_BOTTLENECK_CYCLE_002.json')
        items = {r['id']: r for r in previous['items']}
        items['B1'].update(status='SIGNED_SCALE_HAS_LOCAL_SIGNAL_NOT_OVERALL_GAIN',
            evidence='G2_FULL posthoc observed-scale-decrease delta+.0187962 with pointwise interval[+.0056643,+.0325647]; overall delta-.0008013. These are diagnostic subsets, not an inference gate.',
            next_test='Calibrated final classifier scale and contamination metrics; preserve sparse subgroup uncertainty')
        items['B2'].update(status='PAIRED_SYNTHETIC_INTERVALS_COMPLETE',
            evidence='2000 fixed whole-seed draws completed. Censored recovery comparisons use the common observable seed intersection after a tested reporting repair.',
            next_test='Exact excess-latch trajectories and final detector calibrated alarm comparison')
        items['B3'].update(status='MEMORY_MECHANICS_PASS_DISCRIMINATION_UNRESOLVED',
            evidence='Recovery/persistence works; G2_EVENT all-time delta-.00104617. G2 event effect remains negative in each tested positive break-age bin.',
            next_test='Qualified versus removed raw-memory exact pair decomposition and evidence provenance')
        items['B4'].update(status='ANYTIME_FALSE_LATCH_REMAINS_UNCONTROLLED', severity='HIGH',
            evidence='Default-H G2 reduces long-null latch, but H-transport rates remain roughly.22 to.39 across tested lengths/backgrounds; no nominal5% or anytime interpretation.',
            next_test='Separate fixed-horizon threshold undercoverage from early crossing retention')
        items['B5'].update(status='TOP_PRIORITY_CONDITIONAL_REFERENCE_UNCERTAINTY', severity='CRITICAL',
            evidence='More H did not monotonically reduce false memory. E.g. N0 G2 long-null rates H1000 .349609, H2048 .345703, H5000 .361328 on identical online paths.',
            next_test='Registered512H by512 existing-innovation crossed null diagnostic; causal continuation and diagonal parity required')
        items['B6'].update(status='ALL_FIVE_LEARNED_RECIPES_CLOSED_NO_PROMOTION',
            evidence='G3 pairwise objective delta-.000905190720, 95%CI[-.002705798827,+.001067945057], Ppositive.179;1/5 positive folds. Changing this loss did not rescue the fixed features.',
            next_test='Exact versus sampled training gradients, final runtime and model redundancy; no new learned recipe')
        items['B7'].update(status='BUDGET_ENFORCED_NO_NEW_LEARNED_FAMILY',
            evidence='15 full scientific candidates,5 learned recipes,6 major families,7 screens. One full slot remains but no learned-recipe or family budget remains.',
            next_test='Conditional calibration, forensic, runtime, guard and reproducibility research; do not spend remaining slot on micro-tuning')
        heartbeat = read_json(BASE / 'runtime/HEARTBEAT.json')
        result = {'created_utc': utc_now(), 'cycle': 3, 'status': 'RESEARCH_CONTINUES',
            'wall_seconds': heartbeat['wall_seconds'], 'productive_seconds': heartbeat['productive_seconds'],
            'priority': ['B5', 'B4', 'B3', 'B6', 'B2', 'B1', 'B7'], 'items': list(items.values()),
            'current_primary': 'FROZEN_BL80', 'learned_candidate_result': 'NO_PROMOTION',
            'research_duration_requirement_not_yet_met': True,
            'prior_cycle_sha256': digest(BASE / 'bottlenecks/NEXT7_BOTTLENECK_CYCLE_002.json'),
            'conditional_diagnostic_lock_sha256': digest(BASE / 'CONDITIONAL_NULL_FORENSIC_LOCK.json')}
        save('bottlenecks/NEXT7_BOTTLENECK_CYCLE_003.json', result)
        save('runtime/BOTTLENECK_BOARD_CURRENT.json', result, False)
        history = BASE / 'BOTTLENECK_HISTORY_NEXT7.csv'
        with history.open(newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        rows += [{'cycle': 3, 'created_utc': result['created_utc'], 'id': r['id'], 'status': r['status'],
            'severity': r['severity'], 'evidence': r['evidence'], 'next_test': r['next_test']} for r in items.values()]
        csv_write('BOTTLENECK_HISTORY_NEXT7.csv', rows, immutable=False)
        save('runtime/CURRENT_RESEARCH_STATUS.json', {'best': 'I4_EVENT_SCALE_TREE',
            'best_mean_delta': -.0003624879784099866, 'bottleneck': 'B5_CONDITIONAL_H_UNCERTAINTY',
            'next_task': 'Fused parity then crossed fixed-innovation null forensic, final classifier robustness and runtime',
            'screen_count': 7, 'full_count': 15, 'created_utc': utc_now()}, False)
    note('Bottleneck cycle003 complete', 'All learned families closed; conditional null calibration research continues', 'F4_NULL_GROWTH_CORRECTION')


if __name__ == '__main__':
    main()
