"""Run the full test suite in the final buffer with a dedicated evidence path."""
import os, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
env=os.environ.copy()
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
if __name__=='__main__':
    raise SystemExit(subprocess.call([sys.executable,'-m','pytest','-q','--junitxml=artifacts/validation/pytest_final_real.xml'],cwd=ROOT,env=env))
