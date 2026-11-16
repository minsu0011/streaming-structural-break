"""Same fixed CV latency policy for the independently verified log-energy bank."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.runtime_benchmark import benchmark_runtime
from src.next.log_energy_runtime import PreparedLogEnergyPredictor, source_hash


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', default='Z001_LOG_ENERGY_CURRENT')
    parser.add_argument('--tag', default='research_v1')
    args = parser.parse_args()
    benchmark_runtime(ROOT, args.candidate, 'log_energy_fast_h', args.tag, PreparedLogEnergyPredictor,
        lambda model: {'runtime_source_sha256': source_hash()}, Path(__file__),
        f'artifacts/next/runtime_parity/{args.candidate}/log_energy_fast_h/RESULTS.json')
