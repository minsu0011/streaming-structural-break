"""Verify split feature/tree calls against every original NEXT OOF point."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import sys
import argparse
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.runtime_components import PreparedComponents,source_hash
from src.next.runtime_verification import verify_runtime


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate',default='V007_VARIANCE_FORECAST')
    args = parser.parse_args()
    verify_runtime(ROOT,args.candidate,'separate_feature_tree_calls',PreparedComponents,source_hash,Path(__file__))
