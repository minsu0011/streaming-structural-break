"""Same fixed CV latency policy for the independently verified serial-variance bank."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.runtime_benchmark import benchmark_runtime
from src.next.serial_variance_runtime import PreparedSerialVariancePredictor, source_hash


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', default='J102_SERIAL_VARIANCE_512')
    parser.add_argument('--tag', default='research_v1')
    args = parser.parse_args()
    benchmark_runtime(ROOT, args.candidate, 'serial_variance_fast_h', args.tag, PreparedSerialVariancePredictor,
        lambda model: {'runtime_source_sha256': source_hash()}, Path(__file__),
        f'artifacts/next/runtime_parity/{args.candidate}/serial_variance_fast_h/RESULTS.json')
