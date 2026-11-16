"""NEXT8-only process environment; inherited CPU affinity is unchanged."""
import json
import os
from pathlib import Path
_root = Path(__file__).resolve().parents[2]
_locator = json.loads((_root / 'artifacts/next8/CURRENT_RUN.json').read_text(encoding='utf-8'))
_run = os.environ.get('NEXT8_RUN_ID', _locator['run_id'])
if not _run.startswith('run_') or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_' for c in _run):
    raise ValueError('Invalid NEXT8 run identifier')
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['NUMBA_CACHE_DIR'] = str(_root / 'artifacts/next8' / _run / 'runtime/numba_cache')
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
