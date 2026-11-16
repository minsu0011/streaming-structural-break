"""Frozen performance science with the bounded E003 inference-only runtime."""
import os
import sys
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
from src.next.readonly_bootstrap import install
install()
from src.next3.lean_inference import infer, train

INFER_PARALLELISM = 1
