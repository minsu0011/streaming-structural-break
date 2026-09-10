"""Append-only major-stage checkpoints and minimum-duration enforcement."""
from pathlib import Path
from datetime import datetime, timezone
import json
import subprocess
from src.utils.artifacts import sha256, write_json, utc_now


def best_completed_primary(root, stage_anchor=None):
    records = []
    for path in (Path(root)/'artifacts/next/experiments').glob('*/RESULTS.json'):
        record = json.loads(path.read_text(encoding='utf-8'))
        full = record.get('full') or {}
        if record.get('status') != 'BUG' and full.get('status') == 'COMPLETE':
            records.append((float(full['mean']), path.parent.name))
    records.append((.611292686371357, 'FROZEN_PRIMARY_REFERENCE'))
    # An anchor describes this stage only and is retained separately. Global
    # best is obtained from completed full-CV records, never screen estimates.
    return max(records)


def append_exclusive_checkpoint(directory, result):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    existing = list(directory.glob('STAGE_NEXT_CHECKPOINT_*.json'))
    index = max([int(p.stem.rsplit('_', 1)[1]) for p in existing] or [0])+1
    body = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    while True:
        path = directory/f'STAGE_NEXT_CHECKPOINT_{index:03d}.json'
        try:
            with path.open('x', encoding='utf-8') as stream:
                stream.write(body)
            return path
        except FileExistsError:
            index += 1


def checkpoint(root, current, completed, next_queue, *, failures=None, artifacts=None, best=None):
    root = Path(root)
    session = json.loads((root/'artifacts/next/SESSION_NEXT.json').read_text(encoding='utf-8'))
    started = datetime.fromisoformat(session['research_start_utc'])
    elapsed = (datetime.now(timezone.utc)-started).total_seconds()
    directory = root/'checkpoints'
    global_best, global_id = best_completed_primary(root, best)
    result = {'timestamp': utc_now(), 'elapsed_research_seconds': elapsed,
              'phase': 'RESEARCH', 'current_candidate': current,
              'best_primary_cv_mean': global_best, 'best_primary_cv_candidate_id': global_id,
              'reported_stage_anchor_mean': best, 'completed_experiments': completed,
              'next_queue': next_queue, 'failures': failures or [],
              'artifact_sha256': {p: sha256(root/p) for p in (artifacts or [])},
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
              'minimum_research_seconds': session['minimum_research_seconds'],
              'research_minimum_met': elapsed >= session['minimum_research_seconds'],
              'seal_opened': {k: False for k in 'ABCD'}, 'reduced_new_usage': 0, 'public_submissions': 0}
    return append_exclusive_checkpoint(directory, result)
