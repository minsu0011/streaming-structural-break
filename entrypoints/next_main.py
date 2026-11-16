"""Candidate NEXT entry; the previous main.py deployment remains preserved."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import sys
sys.dont_write_bytecode = True
for _thread_variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_thread_variable]='1'
from src.next.inference import train,infer
INFER_PARALLELISM = 1
