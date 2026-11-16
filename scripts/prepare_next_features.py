import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.data import prepare_dev
from src.next.cache import build_engine_cache
from src.next.engine import EngineConfig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--order', type=int, default=8)
    parser.add_argument('--normalization', default='sd')
    parser.add_argument('--scope', choices=['screen','full'], default='screen')
    args = parser.parse_args()
    data = prepare_dev(ROOT)
    path = build_engine_cache(data, EngineConfig(order=args.order, normalization=args.normalization), args.scope)
    print(path, flush=True)
