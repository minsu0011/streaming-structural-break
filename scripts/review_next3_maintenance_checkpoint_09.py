"""Record actual figure inspection and the now-verified offline setup path."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, atomic_json, digest, progress, read_json, utc_now
from src.next3.bottlenecks import record_board
from src.next3.final_validation import prepared_table_proof, successful_receipt


def main():
    prepared_table_proof(ROOT, BASE/'final_tables/prepared_v2')
    figure = BASE/'figures/engineering_evidence_v1'
    proof = successful_receipt(figure/'RESULTS.json')
    for name, sha in proof['files_sha256'].items(): assert digest(figure/name) == sha
    atomic_json(BASE/'maintenance/ENGINEERING_FIGURE_VISUAL_REVIEW.json', {
        'status': 'PASS', 'created_utc': utc_now(),
        'reviewed_png_sha256': digest(figure/'01_runtime_cost_evidence.png'),
        'reviewer': 'Same agent; actual displayed PNG inspected with view_image',
        'review': 'All three panels, measured trial dots, units, startup target failure and no-throughput-improvement caveat are legible. Cold ratio-of-medians and direct median-paired-ratio boundaries are stated.',
        'original_generation_receipt_sha256': digest(figure/'RESULTS.json')}, immutable=True)
    selection = BASE/'maintenance/PINNED_WHEEL_CACHE_SELECTION.json'
    selected = successful_receipt(selection)
    cache = BASE/'maintenance/pinned_wheel_cache_v1/verification_v2/RESULTS.json'
    cache_proof = successful_receipt(cache)
    assert cache_proof['pin_count'] == 63
    board = read_json(BASE/'bottlenecks/BOTTLENECK_BOARD.json')['bottlenecks']
    sources = [selection, cache, BASE/'maintenance/methodology_review/CURRENT_REPORT.json',
               BASE/'final_audit/FINAL_RUNTIME_PROTOCOL.json',
               BASE/'maintenance/ENGINEERING_FIGURE_VISUAL_REVIEW.json']
    for item in board:
        if item['id'] == 'B4':
            item.update(status='ACTIVE', next_test='Build Korean 44-answer reports and independently cross-check literal numeric claims. Ranking-loss methodology now distinguishes finite surrogate mismatch from population consistency; no new science is authorized.')
        elif item['id'] == 'B8':
            item.update(status='ACTIVE', next_test='Complete final publication/audit and portable reproduction tooling. All 63 local wheel files passed independent metadata, PyPI SHA/size, tag and CRC checks; final raw training and final 30M public API verification remain pending.')
        else:
            assert item['status'] != 'ACTIVE'
            continue
        for path in sources:
            item['evidence'].append({'path': path.relative_to(ROOT).as_posix(), 'sha256': digest(path)})
    record_board(board, 'Offline wheel cache recovered by a top-level metadata schema correction; original failure preserved. Eight result figures have now been visually inspected. Final publication and runtime protocols are prepared, not yet executed; scientific roles remain frozen.')
    atomic_json(BASE/'maintenance/CHECKPOINT_09_COMPLETION.json', {
        'status': 'PASS', 'created_utc': utc_now(), 'scientific_fits': 129,
        'new_fits': 0, 'last_full_test_suite_passes': 753,
        'new_targeted_tests_since_that_full_suite': 53,
        'expected_full_suite_count_is_not_a_pass_claim': True,
        'new_scientific_promotions': 0, 'verified_local_wheel_files': 63,
        'final_raw_training_performed': False, 'final_API_30M_performed': False,
        'source_sha256': digest(Path(__file__)),
        'evidence_sha256': {path.relative_to(ROOT).as_posix(): digest(path) for path in sources}}, immutable=True)
    progress('NEXT3 checkpoint 09', 'Eight figures inspected; offline cache and final runtime protocol reviewed; implementing final Korean claims and audit')
    print('NEXT3 CHECKPOINT 09 PASS; FINAL AUDIT STILL PENDING', flush=True)


if __name__ == '__main__': main()
