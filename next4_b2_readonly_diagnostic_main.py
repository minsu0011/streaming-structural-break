"""Read-only import adapter for the unchanged, unpromoted B2 diagnostic core."""
import os
import sys
sys.dont_write_bytecode=True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from src.next.readonly_bootstrap import install
install()
from src.next4.b2_inference import infer

INFER_PARALLELISM=1
__all__=['infer']
