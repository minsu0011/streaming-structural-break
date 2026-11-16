"""Check exact two-channel runtime specialization on every DEV OOF point."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import sys
import argparse
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.runtime_verification import verify_runtime
from src.next.pruned_variance_runtime import PreparedPrunedVariancePredictor,source_hash


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate',default='V007_VARIANCE_FORECAST')
    args = parser.parse_args()
    verify_runtime(ROOT,args.candidate,'pruned_variance_resident',PreparedPrunedVariancePredictor,source_hash,Path(__file__))
