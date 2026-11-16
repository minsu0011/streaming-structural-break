"""Durable NEXT4-only evidence and observed work signals."""
from pathlib import Path
import os
from src.next3.artifacts import atomic_json, digest, read_json, utc_now

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'artifacts/next4'


def progress(task, detail='', *, phase='RESEARCH', counts_as_research=True):
    if phase not in ('RESEARCH', 'PREFLIGHT', 'FINAL_AUDIT', 'DELIVERY'):
        raise ValueError('Explicit NEXT4 work phase required')
    atomic_json(BASE/'runtime/ACTIVITY.json', {'timestamp':utc_now(), 'pid':os.getpid(),
        'task':str(task), 'detail':str(detail), 'phase':phase,
        'counts_as_research':bool(counts_as_research and phase in ('RESEARCH','PREFLIGHT'))})
