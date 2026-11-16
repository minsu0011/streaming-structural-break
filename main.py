"""Crunch entry point. Supervised research/training source: scripts/run_research.py."""
import os
# Explicit thread limits before numerical imports; no nested worker oversubscription.
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
from src.streaming.inference import train,infer
INFER_PARALLELISM=1
