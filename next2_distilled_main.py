"""Exact compact J102 recipe with its original scientific model definition."""
import os
import sys
sys.dont_write_bytecode=True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from src.next.readonly_bootstrap import install
install()
from src.next2.inference import infer

INFER_PARALLELISM=1


def train(datasets,model_directory_path):
    from src.next2.deployment_training import train_role
    return train_role(datasets,model_directory_path,'DISTILLED')
