"""Original five-fold serial-variance OOF verification through the exact adapter."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.runtime_verification import verify_runtime
from src.next.serial_variance_runtime import PreparedSerialVariancePredictor, source_hash


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', default='J102_SERIAL_VARIANCE_512')
    args = parser.parse_args()
    verify_runtime(ROOT, args.candidate, 'serial_variance_fast_h', PreparedSerialVariancePredictor, source_hash, Path(__file__))
