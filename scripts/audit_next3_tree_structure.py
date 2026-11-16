"""Audit actual frozen forest structure and deterministic artificial boundaries."""
from pathlib import Path
import argparse
import csv
import os
import sys
import time
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, atomic_json, digest, progress, read_json, utc_now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', default='maintenance_v2')
    parser.add_argument('--model-dir', default='artifacts/next3/canonical_deployment/PERFORMANCE_lean_shared_v1')
    args = parser.parse_args()
    if not args.label.isidentifier(): raise ValueError('Unique simple audit label required')
    directory = (ROOT/args.model_dir).resolve()
    if not directory.is_relative_to(BASE.resolve()): raise ValueError('Explicit NEXT3 model directory required')
    final = args.label == 'final_audit'; timing = None
    if final:
        from scripts.prepare_next3_independent_reproduction import time_gate
        timing = time_gate()
        reproduced = read_json(BASE/'independent_reproduction/RESULTS.json')
        assert reproduced['status'] == 'PASS' and reproduced['raw_reproduction_fits'] == 3
        assert directory == (BASE/'independent_reproduction/PERFORMANCE/model').resolve()
    protocol_path = BASE/'maintenance/TREE_STRUCTURAL_AUDIT_PROTOCOL.json'
    lock_path = BASE/'maintenance/TREE_STRUCTURAL_AUDIT_IMPLEMENTATION_LOCK_CORRECTION_001.json'
    protocol = read_json(protocol_path); lock = read_json(lock_path)
    assert lock['protocol_sha256'] == digest(protocol_path)
    for name, sha in lock['source_sha256'].items(): assert digest(ROOT/name) == sha
    assert protocol['new_scientific_fits'] == protocol['new_quality_scores'] == protocol['new_runtime_families'] == 0
    out = BASE/'maintenance'/('tree_structural_'+args.label); out.mkdir(parents=True, exist_ok=False)
    frozen = read_json(BASE/'preflight/FROZEN_REFERENCES.json')['roles']['FROZEN_PERFORMANCE']
    reference_path = ROOT/frozen['model_directory']/'model.sbnext2'
    model_path = directory/'model.sbnext2'
    assert digest(model_path) == digest(reference_path) == protocol['canonical_PERFORMANCE_sha256']
    atomic_json(out/'START.json', {'created_utc':utc_now(), 'model_directory':directory.relative_to(ROOT).as_posix(),
        'model_sha256':digest(model_path), 'reference_model_sha256':digest(reference_path),
        'protocol_sha256':digest(protocol_path), 'implementation_lock_sha256':digest(lock_path),
        'final_audit':final, 'clock':timing, 'new_fits_planned':0}, immutable=True)
    active = [True]; forbidden = []; network = []
    def audit(event, arguments):
        if not active[0]: return
        if event in ('socket.connect','socket.connect_ex'):
            network.append(event); raise RuntimeError('No network in the structural audit')
        if event == 'open':
            raw = arguments[0]
            if not isinstance(raw, (str,bytes,os.PathLike)): return
            path = Path(os.fsdecode(raw)).resolve()
            if path.is_relative_to(ROOT/'data') or path.suffix.lower() in ('.npy','.npz','.parquet'):
                forbidden.append(str(path)); raise RuntimeError('No observation, label, feature cache or saved prediction arrays in this audit')
    sys.addaudithook(audit); began = time.perf_counter()
    try:
        # Install the unchanged public bootstrap before any internal numeric modules.
        import next3_lean_performance_main  # noqa: F401
        import numpy as np
        from src.next2.canonical_model import load as load_original
        from src.next3.lean_model import load as load_lean
        from src.next3.lean_shared_runtime import PreparedLeanSharedPredictor
        from src.models.compact_trees import tree_margins
        from src.next3.packed_trees import packed_tree_margins
        from src.next3.tree_certificate import artificial_probes, certify_nodes
        original = load_original(reference_path); record = load_lean(model_path)
        prepared = PreparedLeanSharedPredictor(record, resident=False)
        primary_columns = np.asarray([original.primary.names.index(name) for name in prepared.primary.names], dtype=np.int32)
        positions = list(prepared.primary.old_positions)+list(prepared.primary.new_positions)
        assert sorted(int(value) for value in positions) == list(range(len(primary_columns)))
        assert np.array_equal(primary_columns[prepared.primary.old_positions], prepared.primary.old_columns)
        assert np.array_equal(primary_columns[prepared.primary.new_positions]-51, prepared.primary.new_columns)
        assert tuple(record['complement']['names']) == tuple(original.complement.names)
        summaries = []; all_records = []; comparisons = 0
        for role, component, packed, columns in (
            ('primary',original.primary,prepared.primary.tree_args,primary_columns),
            ('complement',original.complement,prepared.complement.tree_args,np.arange(len(original.complement.names),dtype=np.int32))):
            structure = certify_nodes(component.compact.args(), component.columns, packed, columns)
            for dtype in (np.float32,np.float64):
                matrix, index = artificial_probes(component.compact.args(),component.columns,len(component.names),dtype)
                expected = tree_margins(np.ascontiguousarray(matrix[:,component.columns]),*component.compact.args())
                actual = packed_tree_margins(np.ascontiguousarray(matrix[:,columns]),*packed)
                if not np.isfinite(expected).all() or not np.isfinite(actual).all() or expected.tobytes() != actual.tobytes():
                    raise RuntimeError('Artificial full-forest margin bytes differ: '+role+'/'+np.dtype(dtype).name)
                comparisons += len(matrix)
                all_records.extend({'component':role,'dtype':np.dtype(dtype).name,**item} for item in index)
                summary = {'component':role, 'dtype':np.dtype(dtype).name, **structure,
                    'attempted_probes':len(index), 'executed_probes':len(matrix),
                    'omitted_unreachable_probes':sum(not row['reachable_for_this_dtype'] for row in index),
                    'leaf_paths_attempted':sum(row['case_kind'] == 'LEAF_PATH' for row in index),
                    'leaf_paths_reached':sum(row['case_kind'] == 'LEAF_PATH' and row['reachable_for_this_dtype'] for row in index),
                    'full_forest_margin_bytes_exact':True, 'max_abs_margin_difference':0.,
                    'original_margin_sha256':__import__('hashlib').sha256(expected.tobytes()).hexdigest()}
                summaries.append(summary)
                progress('Actual forest structure and boundary audit', role+'/'+np.dtype(dtype).name+': '+str(len(matrix))+' artificial full-forest margins exact', phase='FINAL_AUDIT' if final else 'ACTIVE_RESEARCH')
                print('NEXT3 TREE CERTIFICATE',role,np.dtype(dtype).name,len(matrix),'margins exact',flush=True)
        assert not forbidden and not network
    finally:
        active[0] = False
    csv_path = out/'ARTIFICIAL_PROBE_INDEX.csv'
    with csv_path.open('x',encoding='utf-8',newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_records[0])); writer.writeheader(); writer.writerows(all_records)
    for name, sha in lock['source_sha256'].items(): assert digest(ROOT/name) == sha
    assert digest(model_path) == digest(reference_path) == protocol['canonical_PERFORMANCE_sha256']
    atomic_json(out/'RESULTS.json', {'status':'PASS_TREE_STRUCTURE_AND_ARTIFICIAL_BOUNDARIES', 'created_utc':utc_now(),
        'final_audit':final, 'clock':timing, 'model_directory':directory.relative_to(ROOT).as_posix(),
        'canonical_model_sha256':digest(model_path), 'forest_components':2, 'component_dtype_results':summaries,
        'unique_model_nodes_checked':sum(item['nodes'] for item in summaries if item['dtype'] == 'float32'),
        'artificial_full_forest_margin_comparisons':comparisons,
        'all_live_fields_column_maps_and_evaluated_margin_bits_exact':True,
        'primary_old_and_new_output_selectors_match_scientific_names':True,
        'unreachable_cases_retained_not_counted_as_executed':True,
        'new_fits':0, 'new_quality_scores':0, 'new_runtime_families':0,
        'competition_data_or_saved_array_access_attempts':forbidden, 'network_attempts':network,
        'protocol_sha256':digest(protocol_path), 'implementation_lock_sha256':digest(lock_path),
        'files_sha256':{path.name:digest(path) for path in (out/'START.json',csv_path)},
        'source_sha256':lock['source_sha256'], 'elapsed_seconds_not_latency_benchmark':time.perf_counter()-began,
        'scope':'The two fixed numeric forests, selected-column/feature-output mappings and deterministic artificial tree-engine probes. Artificial feature vectors may be unreachable from real causal H/O states. No compiler-formal, hidden-test, robustness or quality guarantee.'}, immutable=True)
    print('NEXT3 TREE STRUCTURAL AUDIT PASS',comparisons,'ARTIFICIAL MARGIN COMPARISONS',flush=True)


if __name__ == '__main__': main()
