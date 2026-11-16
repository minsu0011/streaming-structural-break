"""Freeze fair scalar, initialization and bounded-state throughput workloads."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.next7.common import BASE, save, digest, utc_now, note


def main():
    save('LATENCY_WORKLOAD_LOCK_NEXT7.json', {'status': 'FROZEN', 'created_utc': utc_now(),
        'classifier_methods': ['M0', 'M3', 'M_EVENT', 'G2_EVENT', 'I4_EVENT_SCALE_TREE', 'G2_FULL', 'G3_PAIRWISE_FULL'],
        'state_methods': ['FRESH_CORE20', 'OBSERVABLE_FUSED27', 'G1_READABLE_STATE30', 'G2_FUSED_STATE45'],
        'fitted_fold': 0, 'model_policy': 'Exact original frozen fold0 artifacts and existing NEXT7 heads. No refitting, compiler-specific scientific changes or role reassignment.',
        'workload': {'case': 'N2', 'seed': 63, 'partition': 37, 'control': True, 'H_length': 2048, 'online_length': 4096,
            'origin': 'Exact existing NEXT6 extension benchmark workload.'},
        'warm': {'repetitions': 7, 'rotation': 'Cyclic method order per repetition',
            'timing': 'Actual predict_one API via perf_counter_ns; median/p95/p99 per call. Fresh state each repetition. No output validation inside the timed call beyond the API own validation.',
            'initialization': 'Report model preparation and H initialization separately.'},
        'cold': 'Separate process and empty owned NumBa cache per method. Record model load/prepare, cold H initialization and first scalar separately.',
        'H_length_initialization': 'Use fixed N2 partition33 seed0 H5000 suffixes1000/1500/2048/3000/5000, three repeats, with already-warmed JIT.',
        'direct_throughput': {'points_per_classifier_per_workload': 10000000,
            'workloads': ['LONG_SINGLE_STATE', 'REPEATED_FRESH_SERIES'],
            'long_single_state': 'Repeat the fixed4096-value stream cyclically through one state. This measures10M-step bounded state and counter/qualification support behavior; no scientific detection claim for periodic data.',
            'fresh_series': 'Reset from the same H every4096 values. Measure online API time and H initialization separately and jointly; this exposes per-series calibration cost.',
            'method_rotation': 'Cycle classifier methods in fixed chunks of at most409600 scalar calls; maintain one long state per classifier for LONG_SINGLE_STATE.',
            'timing': 'One timer per chunk, exact call counts, last value and scalar sum checksums. No per-point clock overhead in throughput totals.',
            'memory': 'Verify reachable numeric-array byte counts do not grow between first and final chunks, and record RSS separately.',
            'state_only_10M': 'Report estimates from measured block throughput, explicitly not direct10M state-only runs.'},
        'resources': 'Only after all heavy owned scientific and diagnostic jobs finish. One math-library thread per process. Report actual hardware/system load; do not terminate unrelated user processes.',
        'phase': 'FINAL_AUDIT only, after36000 productive research seconds and concrete research closure admission.',
        'decision': 'Report median ratio versus matched M0 and the existing3x feasibility gate independently of predictive promotion; no positive role from speed alone.',
        'numerical_units': 'Scalar scores are uncalibrated ranking outputs; state arrays are evidence and operational states.',
        'memory_units': 'Report reachable numeric arrays excluding immutable trees/correction parameters and separately bounded mutable state; Python object overhead and shared array accounting limitations explicit.',
        'justification_for_two_direct_workloads': 'Long continuation checks bounded state and behavior beyond16384 supported qualification points; repeated fresh series checks the initialization cost hidden by a single very long stream.',
        'new_scientific_candidates': 0, 'source_sha256': digest(__file__),
        'research_bundle_manifest_sha256': digest(BASE / 'RESEARCH_BUNDLES_EXPORTED.json')})
    note('Final latency workloads frozen', 'Separate scalar tails, H initialization, long-state bounds and repeated-series cost', 'REPRODUCTION_AND_RUNTIME')


if __name__ == '__main__':
    main()
