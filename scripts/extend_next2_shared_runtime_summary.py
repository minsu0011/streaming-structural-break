"""Append the prospectively accepted exact shared runtime to the core and precompute trials."""
from pathlib import Path
import sys
import json
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.utils.artifacts import write_json, sha256, utc_now


def main():
    path = ROOT/'artifacts/next2/latency/BL80_SHARED_V1/RESULTS.json'
    r = json.loads(path.read_text()); assert r['status'] == 'PASS' and r['engineering_replacement_eligible']
    base_path = ROOT/'artifacts/next2/latency/BL80_COMPACT/RESULTS.json'
    assert sha256(base_path) == r['identity']['baseline_receipt_sha256']
    base = json.loads(base_path.read_text())
    for name, digest in r['files_sha256'].items(): assert sha256(path.parent/name) == digest
    for name, digest in r['identity']['proofs_sha256'].items(): assert sha256(ROOT/name) == digest
    frame = pd.read_csv(ROOT/'RUNTIME_PRUNING_RESULTS.csv', float_precision='round_trip')
    frame = frame[frame.runtime_variant != 'BL80_SHARED_V1'].copy()
    assert len(frame) == 11
    row = frame[frame.runtime_variant == 'BL80_COMPACT'].iloc[0].to_dict()
    row.update(runtime_variant='BL80_SHARED_V1', runtime='shared_blend_v1', comparison_runtime='BL80_COMPACT',
        materialized_tree_input_features=72, materialized_feature_reduction_fraction=1-72/101,
        quality_semantics_unchanged=True,
        private_per_series_numeric_bytes=r['numerical_memory']['private_per_series_numeric_bytes'],
        state_array_bytes_including_shared_selectors=r['state_array_bytes_max'],
        model_and_prepared_shared_numeric_bytes=r['numerical_memory']['model_and_prepared_shared_numeric_bytes'],
        private_state_reduction_fraction=1-r['numerical_memory']['private_per_series_numeric_bytes']/row['private_per_series_numeric_bytes'],
        legacy_property_state_reduction_fraction=1-r['state_array_bytes_max']/base['state_array_bytes_max'],
        point_median_us=r['point_distribution']['median_us'], point_p95_us=r['point_distribution']['p95_us'],
        point_p99_us=r['point_distribution']['p99_us'],
        point_median_reduction_fraction=1-r['point_distribution']['median_us']/base['point_distribution']['median_us'],
        historical_init_median_us=r['historical_initialization']['median_us'],
        historical_init_p95_us=r['historical_initialization']['p95_us'], historical_init_p99_us=r['historical_initialization']['p99_us'],
        historical_init_median_reduction_fraction=1-r['historical_initialization']['median_us']/base['historical_initialization']['median_us'],
        feature_update_median_us=r['feature_state_update']['median_us'], model_prediction_median_us=r['model_prediction']['median_us'],
        h_plus_online_amortized_us_per_point=r['h_plus_online_amortized_us_per_point'],
        amortized_reduction_fraction=1-r['h_plus_online_amortized_us_per_point']/base['h_plus_online_amortized_us_per_point'],
        direct_1M_seconds=r['continuous_stream']['measured_first_million_seconds'],
        direct_10M_seconds=r['continuous_stream']['measured_prediction_seconds'],
        continuous_10M_reduction_fraction=1-r['continuous_stream']['measured_prediction_seconds']/base['continuous_stream']['measured_prediction_seconds'],
        state_growth_after_direct_10M_bytes=0, first_history_plus_jit_seconds=r['first_history_plus_jit_seconds_fresh_process'],
        role_status='EXACT_SHARED_PERFORMANCE_RUNTIME_ENGINEERING_ELIGIBLE_FINAL_AUDIT_PENDING',
        engineering_replacement_eligible=True, replacement_reason='All prospectively fixed exactness, speed and state criteria passed',
        receipt=path.relative_to(ROOT).as_posix())
    pd.concat([frame, pd.DataFrame([row])], ignore_index=True).to_csv(ROOT/'RUNTIME_PRUNING_RESULTS.csv', index=False)
    latency = json.loads((ROOT/'LATENCY_NEXT2.json').read_text())
    latency['separate_supplemental_protocols']['BL80_SHARED_V1'] = r
    latency['supplement_note'] = 'Core ten trials unchanged. Precompute trial failed its 10% replacement criterion; shared blend passed its separate prospectively fixed engineering criteria.'
    write_json(ROOT/'LATENCY_NEXT2.json', latency)
    memory = json.loads((ROOT/'STATE_MEMORY_NEXT2.json').read_text())
    memory['separate_supplemental_protocols']['BL80_SHARED_V1'] = {
        **r['numerical_memory'], 'comparison_runtime': 'BL80_COMPACT', 'direct_10M_growth_bytes': 0,
        'receipt_sha256': sha256(path),
        'scope': 'Measured fixed H plus constant state after 10M. Private state and shared prepared-model storage counted separately; Python/JIT/allocator overhead excluded.'}
    write_json(ROOT/'STATE_MEMORY_NEXT2.json', memory)
    out = ROOT/'artifacts/next2/runtime_summary_shared'; out.mkdir(parents=True, exist_ok=True)
    write_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(), 'script_sha256': sha256(Path(__file__)),
        'supplement_receipt_sha256': sha256(path), 'new_measurements': 0, 'new_fits': 0, 'total_runtime_rows': 12,
        'files_sha256': {name: sha256(ROOT/name) for name in ('RUNTIME_PRUNING_RESULTS.csv', 'LATENCY_NEXT2.json', 'STATE_MEMORY_NEXT2.json')}})
    print('NEXT2 SHARED RUNTIME SUMMARY PASS: 12 distinct measured variants', flush=True)


if __name__ == '__main__': main()
