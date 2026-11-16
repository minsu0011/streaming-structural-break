"""Append the separately registered precomputation trial without changing core measurements."""
from pathlib import Path
import sys
import json
import shutil
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.artifacts import write_json, sha256, utc_now


def main():
    out = ROOT/'artifacts/next2/runtime_summary_precomputed'
    out.mkdir(parents=True, exist_ok=True)
    core = json.loads((ROOT/'artifacts/next2/runtime_summary/RESULTS.json').read_text())
    snapshots = out/'CORE10_REPORTS'; snapshots.mkdir(exist_ok=True)
    for name, digest in core['files_sha256'].items():
        target = snapshots/name
        if not target.exists():
            assert sha256(ROOT/name) == digest
            shutil.copyfile(ROOT/name, target)
        assert sha256(target) == digest
    path = ROOT/'artifacts/next2/latency/J_PRECOMPUTED/RESULTS.json'
    r = json.loads(path.read_text()); assert r['status'] == 'PASS'
    base_path = ROOT/'artifacts/next2/latency/J_USED/RESULTS.json'
    assert sha256(base_path) == r['identity']['baseline_receipt_sha256']
    base = json.loads(base_path.read_text())
    for name, digest in r['files_sha256'].items():
        assert sha256(path.parent/name) == digest
    for name, digest in r['identity']['proofs_sha256'].items():
        assert sha256(ROOT/name) == digest
    frame = pd.read_csv(snapshots/'RUNTIME_PRUNING_RESULTS.csv', float_precision='round_trip')
    row = frame[frame.runtime_variant == 'J_USED'].iloc[0].to_dict()
    row.update(runtime_variant='J_PRECOMPUTED', runtime='tree_used_precomputed_serial_v1',
        comparison_runtime='J_USED', private_per_series_numeric_bytes=r['numerical_memory']['private_per_series_numeric_bytes'],
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
        role_status='EXACT_EXPERIMENTAL_RUNTIME_NOT_SELECTED_FOR_REPLACEMENT',
        engineering_replacement_eligible=r['engineering_replacement_eligible'],
        replacement_reason='Direct 10M improvement below the preregistered 10 percent threshold; extra 1200 bytes per state',
        receipt=path.relative_to(ROOT).as_posix())
    # Keyword arguments above read the unmodified baseline row before update.
    frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    frame.to_csv(ROOT/'RUNTIME_PRUNING_RESULTS.csv', index=False)
    latency = json.loads((snapshots/'LATENCY_NEXT2.json').read_text())
    latency['separate_supplemental_protocols'] = {'J_PRECOMPUTED': r}
    latency['supplement_note'] = 'Core ten-variant policy unchanged. Supplemental exact precomputation trial was not selected under its separate frozen replacement criterion.'
    write_json(ROOT/'LATENCY_NEXT2.json', latency)
    memory = json.loads((snapshots/'STATE_MEMORY_NEXT2.json').read_text())
    memory['separate_supplemental_protocols'] = {'J_PRECOMPUTED': {
        **r['numerical_memory'], 'comparison_runtime': 'J_USED', 'additional_private_bytes': 1200,
        'direct_10M_growth_bytes': 0, 'receipt_sha256': sha256(path),
        'scope': 'One directly measured fixed H example plus constant state through 10M points; core three-H ownership table remains separate.'}}
    write_json(ROOT/'STATE_MEMORY_NEXT2.json', memory)
    write_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(),
        'script_sha256': sha256(Path(__file__)), 'core_receipt_sha256': sha256(ROOT/'artifacts/next2/runtime_summary/RESULTS.json'),
        'supplement_receipt_sha256': sha256(path), 'core10_report_snapshot_sha256': core['files_sha256'],
        'files_sha256': {name: sha256(ROOT/name) for name in core['files_sha256']},
        'total_runtime_rows': len(frame), 'new_measurements': 0, 'new_fits': 0})
    print('NEXT2 RUNTIME SUMMARY: ten original variants plus one explicitly unselected supplement', flush=True)


if __name__ == '__main__':
    main()
