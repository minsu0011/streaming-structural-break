"""Freeze a real-DEV latency sample and fair measurement rules before timing."""
from pathlib import Path
import hashlib
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.guard import SealGuard
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    path = ROOT/'configs/next_latency_policy.json'
    if path.exists():
        print('Latency policy already frozen')
        return
    guard = SealGuard(ROOT)
    ids = [sid for sid in guard.dev_ids if guard.split.assignment(sid)[1]==0]
    seed = 202609096
    selected = sorted(ids,key=lambda sid:(hashlib.sha256(f'next-latency-v1:{seed}:{sid}'.encode()).digest(),sid))[:128]
    guard.admit(selected,purpose='latency policy sample')
    write_json(path,{'status':'LOCKED','created_utc':utc_now(),'sample_seed':seed,'series_ids':selected,
        'sample_policy':'128 original-fold-0 DEV IDs chosen by fixed SHA256 order; no values, outcomes or timing results in selection.',
        'split_sha256':guard.split.digest,'seal_lock_sha256':guard.lock_sha,
        'continuous_points':10000000,'block_points':10000,'measured_checkpoint_points':1000000,
        'warmup':'One complete factory initialization and scalar update on the first sampled real H before timed measurements; JIT/load times reported separately.',
        'point_distribution':'Individual scalar-call time for every point in all 128 actual held-out sequences; report median, p95 and p99 without subtracting timer overhead.',
        'primary_point_gate':'Median individual scalar-call microseconds, divided by the contemporaneous frozen PRIMARY reference median under the same protocol.',
        'amortized_gate':'Aggregate measured historical initialization plus real-sequence point-loop wall time, divided by total actual points; compare to the same-protocol PRIMARY value.',
        'continuous_stream':'Repeat a bounded 10000-value buffer formed from the sampled real online prefixes for 10000000 scalar updates on one state. This is a synthetic-duration engineering stress, never a competition performance score.',
        'continuous_outputs':'Validate finite float32 probabilities and fixed state size; record 1M and 10M actual throughput and output digest.',
        'scope':'Local Windows CPU, no socket, cloud or input-file IO in timed intervals. CV fold-0 models for research; repeat selected final-artifact measurements during the additional final audit.',
        'isolation':'Reject timing if another non-ancestor Python process has this workspace as its current directory; check before timing and at each 1M checkpoint.',
        'source_requirement':'Every optimized candidate runtime needs saved full-primary-OOF bitwise parity before its timing can support promotion.',
        'reduced_new_usage':0,'seal_rows':0})
    print('FROZEN LATENCY SAMPLE',len(selected),'series')


if __name__=='__main__':
    main()
