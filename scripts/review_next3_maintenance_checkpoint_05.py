"""Review completed E002/D003 evidence and a static reproduction dependency fix."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, read_json, atomic_json, digest, utc_now, progress
from src.next3.bottlenecks import record_board


def main():
    board = read_json(BASE/'bottlenecks/BOTTLENECK_BOARD.json')['bottlenecks']
    for item in board:
        if item['id'] == 'B8':
            item['next_test'] = 'E002 all-DEV actual public API passed and memory gate met. E003 final runtime family tests inference-only dependency loading; full pinned raw reproduction remains gated to final audit.'
            for relative in ('artifacts/next3/engineering/packed_shared/timing/RESULTS.json', 'artifacts/next3/packed_public_api_all_dev/RESULTS.json'):
                item['evidence'].append({'path': relative, 'sha256': digest(ROOT/relative)})
        if item['id'] == 'B4':
            item['next_test'] = 'Finish existing-cache every-row streaming audits and fixed held-out router metric alignment; no further scientific fits or promotion screening.'
    record_board(board, 'E002 packed runtime satisfies fixed memory gate with unchanged every-point public predictions. Scientific finite menu stays closed. E003 and diagnostic lineage maintenance remain the only active bottlenecks.')
    atomic_json(BASE/'maintenance/REPRODUCTION_STATIC_DEPENDENCY_REPAIR_001.json', {
        'status': 'PRE_EXECUTION_CORRECTION', 'created_utc': utc_now(),
        'finding': 'Initial NEXT2 preparation template omitted eight files that its later PREPARATION_READY repair had added.',
        'corrected_next3_files': ['configs/features.json', 'configs/arch_input.json', 'configs/ar_residual_input.json',
            'configs/historical_context.json', 'configs/historical_quantiles.json', 'configs/rank_gaussian.json',
            'configs/magnitude_dependence.json', 'ROBUSTNESS_SPLIT_LOCK.json'],
        'new_reproduction_executions': 0, 'old_files_modified': 0,
        'next3_prepare_script_sha256': digest(ROOT/'scripts/prepare_next3_independent_reproduction.py'),
    }, immutable=True)
    progress('NEXT3 maintenance checkpoint reviewed', 'B8 dependency-loading optimization and B4 fixed diagnostics; no new science menu')


if __name__ == '__main__': main()
