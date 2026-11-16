"""Original five-fold log-energy OOF verification through the exact adapter."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.runtime_verification import verify_runtime
from src.next.log_energy_runtime import PreparedLogEnergyPredictor, source_hash


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', default='Z001_LOG_ENERGY_CURRENT')
    args = parser.parse_args()
    verify_runtime(ROOT, args.candidate, 'log_energy_fast_h', PreparedLogEnergyPredictor, source_hash, Path(__file__))
