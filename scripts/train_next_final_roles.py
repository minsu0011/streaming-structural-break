"""Audit frozen final roles through the real raw iterator training functions.

Run only after the ten-hour research boundary and role freeze. The delivery
weights must match both cached and independently reconstructed all-DEV fits.
No training-set metric is calculated and no sealed values are supplied.
"""
import os
import sys
sys.dont_write_bytecode = True
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '4'
from pathlib import Path
import argparse
import importlib
import json
import shutil
import subprocess
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.guard import SealGuard
from src.next.data import iter_dev_series, RAW_SHA
from src.next.research_clock import ResearchClock
from src.next.references import verify_reference_sources
from src.utils.artifacts import write_json, sha256, utc_now


class OpaqueSealField:
    def __array__(self, *args, **kwargs):
        raise AssertionError('A sealed array was inspected')
    def __len__(self):
        raise AssertionError('A sealed length was inspected')
    def __float__(self):
        raise AssertionError('A sealed scalar was inspected')
    def __int__(self):
        raise AssertionError('A sealed target was inspected')
    def __bool__(self):
        raise AssertionError('A sealed field was tested')


def raw_iterator(guard):
    yield from iter_dev_series(guard)
    for sid in sorted(guard.seal_ids):
        yield sid, OpaqueSealField(), OpaqueSealField(), OpaqueSealField()


FAMILIES = {
    'conditional_variance': ('next_readonly_main', 'src.next.training', 'configs/next_deployment.json', 'scripts/check_next_artifact_cold_child.py'),
    'log_energy': ('next_registered_main', 'src.next.registered_training', 'configs/next_registered_deployment.json', 'scripts/check_next_registered_cold_child.py'),
    'serial_variance': ('next_serial_main', 'src.next.serial_training', 'configs/next_serial_deployment.json', 'scripts/check_next_serial_cold_child.py'),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', required=True, choices=('PRIMARY_DEV_CANDIDATE', 'ROBUST_DEV_CANDIDATE', 'FAST_DEV_CANDIDATE'))
    args = parser.parse_args()
    clock = ResearchClock(ROOT).snapshot()
    if not clock['minimum_research_met'] or not clock['final_audit_started']:
        raise RuntimeError('Final iterator audit requires ten hours and explicit audit start')
    verify_reference_sources(ROOT)
    guard = SealGuard(ROOT)
    selection_path = ROOT/'artifacts/next/final_selection/RESULTS.json'
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    row = next(row for row in selection['candidates'] if row['role'] == args.role)
    spec_path = ROOT/row['specification']
    if sha256(spec_path) != selection['specification_files_sha256'][row['specification']]:
        raise RuntimeError('Frozen role specification changed')
    spec = json.loads(spec_path.read_text(encoding='utf-8'))
    source = ROOT/row['source_model_directory']
    original = json.loads((source/'model.json').read_text(encoding='utf-8'))
    if sha256(source/'model.joblib') != row['model_sha256'] or sha256(source/'model.json') != row['descriptor_sha256']:
        raise RuntimeError('Verified source artifact changed')
    engineering_path = ROOT/'artifacts/next/final_engineering'/row['experiment_id']/'RESULTS.json'
    engineering = json.loads(engineering_path.read_text(encoding='utf-8'))
    if engineering['status'] != 'PASS' or sha256(engineering_path) != row['engineering_report_sha256']:
        raise RuntimeError('Final engineering receipt changed')
    for relative, digest in engineering['evidence_files_sha256'].items():
        if sha256(ROOT/relative) != digest:
            raise RuntimeError('Engineering evidence changed: '+relative)
    for name, digest in RAW_SHA.items():
        if sha256(ROOT/'data/raw'/name) != digest:
            raise RuntimeError('Canonical raw input changed')
    family = row['candidate']['extension']['kind']
    entry_name, trainer_name, default_name, cold_name = FAMILIES[family]
    entry = importlib.import_module(entry_name)
    trainer = importlib.import_module(trainer_name)
    out = ROOT/'artifacts/next/final_training'/args.role
    delivery = ROOT/'artifacts/next/deployment/final'/args.role
    out.mkdir(parents=True, exist_ok=True)
    identity = {'role': args.role, 'candidate': row['experiment_id'], 'selection_sha256': sha256(selection_path),
        'specification_sha256': sha256(spec_path), 'model_sha256': row['model_sha256'],
        'trainer_sha256': trainer.training_hash(), 'driver_sha256': sha256(Path(__file__)),
        'entry_sha256': sha256(ROOT/(entry_name+'.py')), 'cold_child_sha256': sha256(ROOT/cold_name),
        'raw_input_sha256': RAW_SHA}
    policy_path = out/'POLICY.json'
    if policy_path.exists():
        if json.loads(policy_path.read_text(encoding='utf-8')) != identity:
            raise RuntimeError('Final training audit identity changed; preserve and explicitly revise')
    else:
        write_json(policy_path, identity)
    if (out/'RESULTS.json').exists():
        retained = json.loads((out/'RESULTS.json').read_text(encoding='utf-8'))
        for relative, digest in retained['evidence_files_sha256'].items():
            if sha256(ROOT/relative) != digest:
                raise RuntimeError('Completed final training evidence changed')
        print('FINAL ROLE TRAIN retained PASS', args.role, flush=True)
        return
    tick = time.perf_counter()
    trained = out/'raw_iterator_artifact'
    default_spec = json.loads((ROOT/default_name).read_text(encoding='utf-8'))
    default_entry_used = default_spec == spec
    if not (trained/'model.json').exists():
        if (trained/'model.joblib').exists():
            raise RuntimeError('Partial raw iterator model requires explicit recovery, not another fit')
        if default_entry_used:
            descriptor = entry.train(raw_iterator(guard), str(trained))
        else:
            descriptor = trainer.train_frozen_candidate(raw_iterator(guard), trained, spec, root=ROOT)
    else:
        descriptor = json.loads((trained/'model.json').read_text(encoding='utf-8'))
    checks = {'raw_iterator_model_bytes_match_verified_cache_and_independent_fit': sha256(trained/'model.joblib') == row['model_sha256'],
        'opaque_seal_tuples_skipped': descriptor['seal_series_skipped_before_value_or_label_access'] == 2000,
        'seal_rows_fitted_zero': descriptor['seal_rows_fitted'] == 0,
        'complete_dev_rows': descriptor['training']['training_rows'] == 4014405,
        'complete_dev_series': descriptor['training']['training_series'] == 8000,
        'frozen_role_status': descriptor['status'] == spec['status'],
        'actual_training_source': descriptor['training_source_sha256'] == trainer.training_hash()}
    for name in ('kind', 'runtime', 'candidate', 'model_sha256', 'runtime_sha256', 'inference_sha256'):
        checks['inference_descriptor_'+name+'_matches'] = descriptor[name] == original[name]
    for name in ('feature_matrix_sha256', 'ordered_ids_sha256', 'ordered_targets_sha256', 'retained_columns'):
        checks['raw_training_'+name+'_matches'] = descriptor['training'][name] == original['training'][name]
    if not all(checks.values()):
        write_json(out/'TRAINING_FAILURE.json', {'status': 'FAIL', 'checks': checks})
        raise RuntimeError('Raw final training did not match the verified artifact')
    delivery.mkdir(parents=True, exist_ok=True)
    for name in ('model.joblib', 'model.json'):
        target = delivery/name
        if target.exists():
            if sha256(target) != sha256(trained/name):
                raise RuntimeError('Existing delivery file differs; refusing overwrite')
        else:
            with (trained/name).open('rb') as source_stream, target.open('xb') as target_stream:
                shutil.copyfileobj(source_stream, target_stream)
    original_fixture = ROOT/'data/next/all_dev_artifact_contract'/row['experiment_id']/'research_v1/fixture.json'
    fixture = json.loads(original_fixture.read_text(encoding='utf-8'))
    if fixture['model_sha256'] != row['model_sha256']:
        raise RuntimeError('The previous all-DEV contract fixture has different model bytes')
    fixture['model_directory'] = str(delivery)
    fixture_path = ROOT/'data/next/final_artifact_contract'/args.role/'fixture.json'
    write_json(fixture_path, fixture)
    cold_path = out/'DELIVERY_COLD_IMPORT.json'
    if not cold_path.exists():
        subprocess.run([sys.executable, str(ROOT/cold_name), '--fixture', str(fixture_path), '--output', str(cold_path)], cwd=ROOT, check=True)
    cold = json.loads(cold_path.read_text(encoding='utf-8'))
    checks['delivery_fresh_cold_process_bitwise_sample'] = cold['status'] == 'PASS' and cold['model_sha256'] == row['model_sha256']
    checks['delivery_cold_filesystem_mutations_zero'] = not cold['audit']['filesystem_mutation_attempts']
    guard.check_state()
    if not all(checks.values()):
        raise RuntimeError('Delivery cold contract failed')
    paths = [selection_path, spec_path, policy_path, engineering_path, source/'model.joblib', source/'model.json',
        trained/'model.joblib', trained/'model.json', delivery/'model.joblib', delivery/'model.json', cold_path, ROOT/cold_name, Path(__file__)]
    write_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(), 'identity': identity, 'checks': checks,
        'role': args.role, 'candidate': row['experiment_id'], 'delivery_model_directory': delivery.relative_to(ROOT).as_posix(),
        'model_sha256': sha256(delivery/'model.joblib'), 'delivery_descriptor_sha256': sha256(delivery/'model.json'),
        'default_entry_train_used': default_entry_used, 'training_function': entry_name+'.train' if default_entry_used else trainer_name+'.train_frozen_candidate',
        'elapsed_seconds': time.perf_counter()-tick, 'evidence_files_sha256': {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths},
        'full_api_and_latency_bridge': 'Identical model weights and all inference-relevant descriptor fields connect the complete-DEV 4,014,405-point API and 10-million-point timing audits to this delivery. A fresh cold subprocess additionally checks the delivery path and descriptor.',
        'training_quality_metric_computed': False, 'sealed_observation_or_target_supplied': False,
        'scope': 'The real training implementation consumed 8,000 predicate-filtered DEV series plus 2,000 opaque seal tuples. Canonical raw-byte hashes are integrity checks, not sealed evaluations.'})
    print('FINAL ROLE TRAIN PASS', args.role, row['experiment_id'], flush=True)


if __name__ == '__main__':
    main()
