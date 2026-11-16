"""Record the next characterization task before implementing its interventions."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, read_json, atomic_json, digest, utc_now, progress
from src.next3.bottlenecks import record_board


def main():
    protocol = ROOT/'docs/NEXT3_OUTLIER_STATE_TRANSFER_PROTOCOL.md'
    atomic_json(BASE/'hypotheses/D005_OUTLIER_STATE_TRANSFER.json', {
        'created_utc': utc_now(),
        'HYPOTHESIS': 'The already observed O1 probability persistence can be localized by matched recurrent-state block interventions, beyond descriptive age-boundary association.',
        'BOTTLENECK': 'B6',
        'CHANGE': 'Evaluation-only transfers of six existing recurrent state blocks in the original frozen BL80; same original O1 observations and no candidate model.',
        'EXPECTED_EFFECT': 'Identify which block resets remove or induce probability/alarm persistence under explicit synthetic interventions, without claiming additive or real-DEV causal effects.',
        'RISKS': 'Hybrid states can be unreachable under a single ordinary history; interactions and adaptive diagnostic choice limit causal/generalization claims.',
        'PRIMARY_METRIC': 'Matched probability and fixed-threshold alarm contrasts after each registered state transfer; no TS-AUC on all-null O1.',
        'SECONDARY_DIAGNOSTIC': 'Exact original components/BL predictions, clone and complete-transfer equivalence, no mutable aliasing and fixed numeric state.',
        'STOP_RULE': 'Any invariant fails: stop and preserve evidence. Never fit, tune, promote or change inference from this diagnostic.',
        'protocol_sha256': digest(protocol), 'bootstrap_seed': 2026091071, 'bootstrap_replicates': 2000,
        'anchors': [0, 127, 255, 511], 'new_fits': 0, 'scientific_candidates': 0,
    }, immutable=True)
    items = read_json(BASE/'bottlenecks/BOTTLENECK_BOARD.json')['bottlenecks']
    for item in items:
        if item['id'] == 'B4':
            item['status'] = 'UNCHANGED'
            item['next_test'] = 'Completed fixed-pool uncertainty, every-row scientific cache audits and held-out objective alignment; independent validation limitation remains. No further scientific selection.'
            path = BASE/'router_metric_alignment/RESULTS.json'
            item['evidence'].append({'path': path.relative_to(ROOT).as_posix(), 'sha256': digest(path)})
        if item['id'] == 'B6':
            item['status'] = 'ACTIVE'; item['achievable_improvement'] = .6
            item['next_test'] = 'D005 matched-state interventions on original O1 cohort only; distinguish state persistence from age-window association, without new fits or candidate deployment.'
            item['stop_rule'] = 'Exact clone/full-transfer and original prediction checks required; no inferred remedy or scientific promotion.'
            item['evidence'].append({'path': protocol.relative_to(ROOT).as_posix(), 'sha256': digest(protocol)})
        if item['id'] == 'B8':
            item['next_test'] = 'E003 cold gate passed: 26.35% lower process RSS; finish all-point parity and three paired direct 10M trials. Corrected unit catalog assertion preserved with original failure/source lock.'
            path = BASE/'engineering/lean_shared/cold/RESULTS.json'
            item['evidence'].append({'path': path.relative_to(ROOT).as_posix(), 'sha256': digest(path)})
    record_board(items, 'Early evidence-triggered review: B4 fixed diagnostics completed, B8 cold-memory gate passed. Activate B6 counterexample characterization alongside remaining B8 verification; scientific menu stays closed.')
    figures = BASE/'figures/scientific_menu_v1'; result = read_json(figures/'RESULTS.json')
    for name, sha in result['files_sha256'].items(): assert digest(figures/name) == sha
    atomic_json(BASE/'maintenance/SCIENTIFIC_FIGURE_VISUAL_REVIEW.json', {
        'status': 'PASS', 'created_utc': utc_now(), 'source_receipt_sha256': digest(figures/'RESULTS.json'),
        'reviewed_pngs': [name for name in result['files_sha256'] if name.endswith('.png')],
        'review': 'All three PNGs visually inspected: legible axes, distinct individual/joint intervals, visible gate and caveats, no clipped labels or legend over data.',
        'scope': 'Presentation review only, no new statistical computation or selection',
    }, immutable=True)
    progress('D005 counterexample characterization prospectively registered', 'B8 and B6 only active; fixed original synthetic cohort, no new scientific model')


if __name__ == '__main__': main()
