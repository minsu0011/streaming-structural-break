"""Optional registered-family NEXT entry; original and variance entries persist."""
import os
import sys
sys.dont_write_bytecode = True
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
from src.next.readonly_bootstrap import install
install()
from src.next.registered_inference import train, infer

INFER_PARALLELISM = 1
