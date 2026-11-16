"""NEXT9 process initialization: preserved legacy imports, own short cache."""
import json
import os
from pathlib import Path

# Legacy FeatureState is reused read-only. Initialize its package before setting
# NEXT9 environment so a later legacy import cannot redirect compiled caches.
import src.next8

ROOT = Path(__file__).resolve().parents[2]
_locator = json.loads((ROOT / 'artifacts/next9/CURRENT_RUN.json').read_text(encoding='utf-8'))
_run = os.environ.get('NEXT9_RUN_ID', _locator['run_id'])
if not _run.startswith('run_') or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_' for c in _run):
    raise ValueError('Invalid NEXT9 run ID')
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['NUMBA_CACHE_DIR'] = str(ROOT / 'artifacts/next9' / _run / 'nc')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['PYTHONUTF8'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'
