"""Final same-protocol public API 10M verification using raw-reproduced models."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import csv
import importlib
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.next3.artifacts import BASE, atomic_json, digest, progress, read_json, utc_now
from src.next3.final_runtime_probe import consume_public_probe
from src.next.data import prepare_dev
from scripts.benchmark_next2_runtime import assert_isolated
from scripts.prepare_next3_independent_reproduction import time_gate


def main():
    timing = time_gate(); assert_isolated()
    registration = read_json(BASE/'final_audit/FINAL_RUNTIME_PROTOCOL.json')
    assert registration['status'] == 'REGISTERED_FINAL_VERIFICATION'
    for relative, sha in registration['source_sha256'].items(): assert digest(ROOT/relative) == sha
    reproduction_path = BASE/'independent_reproduction/RESULTS.json'; reproduction = read_json(reproduction_path)
    assert reproduction['status'] == 'PASS' and reproduction['raw_reproduction_fits'] == 3
    policy = read_json(ROOT/'configs/next2/RUNTIME_MEASUREMENT_LOCK.json')
    data = prepare_dev(ROOT); data.guard.admit(policy['series_ids'], purpose='Final fixed public runtime verification')
    examples = [(sid, *data.get(sid)[:2]) for sid in policy['series_ids']]
    historical = examples[0][1]; buffer = np.concatenate([row[2] for row in examples])[:policy['block_points']]
    assert len(buffer) == 10000
    expected_j = read_json(ROOT/'LATENCY_NEXT2.json')['variants']['J_USED']['continuous_stream']['output_sha256']
    expected_p = read_json(ROOT/'artifacts/next2/latency/BL80_SHARED_V1/RESULTS.json')['continuous_stream']['output_sha256']
    planned = [('DISTILLED', 'next2_canonical_distilled_main', BASE/'independent_reproduction/DISTILLED/model', expected_j),
        ('REFERENCE_PERFORMANCE', 'next2_shared_performance_main', ROOT/'artifacts/next2/canonical_deployment/PERFORMANCE_shared_v1', expected_p),
        ('PERFORMANCE', 'next3_lean_performance_main', BASE/'independent_reproduction/PERFORMANCE/model', expected_p)]
    assert [item[0] for item in planned] == registration['fixed_order']
    out = BASE/'final_audit/runtime'; out.mkdir(parents=True, exist_ok=False)
    atomic_json(out/'START.json', {'created_utc': utc_now(), 'clock': timing,
        'protocol_sha256': digest(BASE/'final_audit/FINAL_RUNTIME_PROTOCOL.json'),
        'reproduction_sha256': digest(reproduction_path), 'new_fits': 0}, immutable=True)
    records = {}
    for role, module, directory, expected in planned:
        assert_isolated(); entry = importlib.import_module(module)
        def on_block(record):
            if record['completed_points'] % 1_000_000 == 0:
                assert_isolated()
                progress('Final actual public API runtime', role+' '+str(record['completed_points'])+' verified points', phase='FINAL_AUDIT')
                print('NEXT3 FINAL API', role, record['completed_points'], round(record['cumulative_generator_loop_seconds'], 3), flush=True)
        result = consume_public_probe(entry, directory, historical, buffer,
            points=10_000_000, block_points=10_000, on_block=on_block)
        assert result['output_sha256'] == expected
        blocks = result.pop('blocks'); csv_path = out/(role+'_BLOCKS.csv')
        with csv_path.open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(blocks[0])); writer.writeheader(); writer.writerows(blocks)
        result.update(status='PASS', created_utc=utc_now(), role=role, entry_module=module,
            model_directory=directory.relative_to(ROOT).as_posix(), model_sha256=digest(directory/'model.sbnext2'),
            descriptor_sha256=digest(directory/'model.json'), blocks_sha256=digest(csv_path),
            expected_output_sha256=expected, new_fits=0, new_quality_scores=0, seal_rows=0)
        atomic_json(out/(role+'.json'), result, immutable=True); records[role] = result
    ratio = records['PERFORMANCE']['measured_generator_loop_seconds']/records['REFERENCE_PERFORMANCE']['measured_generator_loop_seconds']
    if ratio > 1.5: raise RuntimeError('Final selected public runtime exceeds fixed 1.5x same-protocol reference bound')
    data.guard.check_state()
    atomic_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(),
        'selected_roles_verified': ['DISTILLED', 'PERFORMANCE'], 'per_entry': records,
        'public_API_output_fingerprints_exact': True, 'one_current_point_per_yield': True,
        'constant_numeric_state': True, 'isolated_from_other_project_python_workers': True,
        'selected_performance_vs_reference_public_API_10M_ratio': ratio, 'fixed_maximum_ratio': 1.5,
        'total_actual_public_API_points': 30_000_000, 'new_fits': 0, 'new_scientific_candidates': 0,
        'seal_rows': 0, 'clock': time_gate(), 'protocol_sha256': digest(BASE/'final_audit/FINAL_RUNTIME_PROTOCOL.json'),
        'files_sha256': {p.name: digest(p) for p in out.iterdir() if p.is_file()},
        'scope': 'One final same-protocol functional/cost verification. It does not replace the earlier three paired engineering trials or claim a cold-start or pure-kernel speed improvement.'}, immutable=True)
    print('NEXT3 FINAL PUBLIC API RUNTIME PASS 30M POINTS; SELECTED/REFERENCE RATIO', ratio, flush=True)


if __name__ == '__main__': main()
