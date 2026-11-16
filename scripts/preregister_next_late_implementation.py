"""Freeze reviewed late-scoring implementation before any real blend scoring."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.late_evaluation import implementation_files, QUEUES
from src.next.research_clock import ResearchClock
from src.utils.artifacts import write_json, sha256, utc_now

if __name__ == '__main__':
    out = ROOT/'configs/next_late_implementation_lock.json'
    if out.exists():
        raise RuntimeError('The late implementation lock is immutable')
    clock = ResearchClock(ROOT).snapshot()
    if clock['research_elapsed_seconds'] >= 28800:
        raise RuntimeError('Preregistration must precede the eighth research hour')
    files = (*implementation_files(), *QUEUES, 'src/validation/group_bootstrap.py',
        'src/scoring/ts_auc.py', 'ROBUSTNESS_SPLIT_LOCK.json')
    write_json(out, {'status': 'LOCKED', 'created_utc': utc_now(),
        'research_elapsed_at_lock': clock['research_elapsed_seconds'],
        'files_sha256': {relative: sha256(ROOT/relative) for relative in files},
        'no_real_combined_predictions_formed': True,
        'first_permitted_combined_score_elapsed_seconds': 28800,
        'queues': list(QUEUES), 'fitting': 'No additional component fits or fitted blend weights',
        'forensic_adapter': 'Explicit replacement of the frozen forensic script OOF loader only. Both adapter and unchanged engine are pinned; all comparisons, support rules, resampling and output schemas remain identical.',
        'marginal_gate': 'Only L101-L103 inherit the separately preregistered +.0005 marginal complexity threshold.',
        'seal_rows': 0, 'reduced_new_uses': 0})
    print('LOCKED LATE IMPLEMENTATION', sha256(out), flush=True)
