"""Run the frozen isolated CV-model benchmark for a reviewed runtime."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.runtime_benchmark import benchmark_runtime,NativeNextPrepared
from src.utils.artifacts import sha256


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate',required=True)
    parser.add_argument('--runtime',choices=['reference','native','fused','pruned','fast_h'],required=True)
    parser.add_argument('--tag',default='research_v1')
    args = parser.parse_args()
    parity = None
    if args.runtime=='reference':
        from src.streaming.prepared import prepare_model,source_hashes
        prepare = lambda model:prepare_model(model,fast_initialization=True)
        receipt = lambda model:source_hashes(model,fast_initialization=True)
    elif args.runtime=='native':
        prepare = NativeNextPrepared
        receipt = lambda model:{'runtime_source_sha256':sha256(ROOT/'src/next/runtime_benchmark.py')}
    elif args.runtime=='fused':
        from src.next.fused_runtime import PreparedNextPredictor,source_hash
        prepare = PreparedNextPredictor
        receipt = lambda model:{'runtime_source_sha256':source_hash()}
        parity = f'artifacts/next/fused_parity/{args.candidate}_resident/RESULTS.json'
    elif args.runtime=='pruned':
        from src.next.pruned_variance_runtime import PreparedPrunedVariancePredictor,source_hash
        prepare = PreparedPrunedVariancePredictor
        receipt = lambda model:{'runtime_source_sha256':source_hash()}
        parity = f'artifacts/next/runtime_parity/{args.candidate}/pruned_variance_resident/RESULTS.json'
    else:
        from src.next.fast_variance_initialization import PreparedFastVariancePredictor,source_hash
        prepare = PreparedFastVariancePredictor
        receipt = lambda model:{'runtime_source_sha256':source_hash()}
        parity = f'artifacts/next/runtime_parity/{args.candidate}/pruned_variance_fast_h/RESULTS.json'
    benchmark_runtime(ROOT,args.candidate,args.runtime,args.tag,prepare,receipt,Path(__file__),parity)
