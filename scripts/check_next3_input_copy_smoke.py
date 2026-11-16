"""Maintenance only: copied-input admission/import checks, no fits or raw data."""
import os
import sys
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--origin', required=True)
    args = parser.parse_args(); origin = Path(args.origin).resolve()
    payload = Path(__file__).resolve().parents[1]; workspace = payload.parent
    environment = Path(sys.prefix).resolve()
    assert not workspace.is_relative_to(origin) and payload.name == 'payload'
    assert not (payload/'data').exists()
    sys.path.insert(0, str(payload))
    plan = json.loads((payload/'SMOKE_INPUT_MANIFEST.json').read_text(encoding='utf-8'))
    for name, expected in plan['files_sha256'].items():
        assert digest(payload/name) == expected
    active = [True]; phase = ['TRAIN_ADMISSION']; denied = []; writes = []; network = []
    def audit(event, arguments):
        if not active[0]: return
        if event in ('socket.connect', 'socket.connect_ex'):
            network.append(event); raise RuntimeError('No network in copied-input smoke')
        if event == 'open':
            raw, mode, flags = arguments
            if not isinstance(raw, (str, bytes, os.PathLike)): return
            decoded = os.fsdecode(raw)
            if os.path.normcase(os.path.normpath(decoded)) == os.path.normcase(os.path.normpath(os.devnull)): return
            path = Path(decoded).resolve()
            writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if path.is_relative_to(origin) and not path.is_relative_to(environment):
                denied.append(str(path)); raise RuntimeError('Origin project read denied; existing environment is explicitly allowed')
            if writing and (phase[0] == 'INFERENCE' or not path.is_relative_to(workspace)):
                writes.append(str(path)); raise RuntimeError('Unexpected persistent write during copy smoke')
        if event in ('os.remove', 'os.rmdir', 'os.mkdir', 'os.chmod', 'os.truncate', 'os.rename', 'os.link', 'os.symlink'):
            if phase[0] == 'INFERENCE':
                writes.append(event); raise RuntimeError('No persistent inference mutation')
    sys.addaudithook(audit)
    began = time.perf_counter(); results = {}; fit_calls = []
    try:
        versions = {}
        for line in (payload/'requirements-lock.txt').read_text().splitlines():
            if not line.strip() or line.startswith('#'): continue
            name, version = line.split('=='); installed = importlib.metadata.version(name)
            assert installed == version; versions[name] = installed
        assert len(versions) == 63
        # Match the real final reproduction: the public entry installs the
        # frozen readonly bootstrap before any project training dispatchers.
        entries = {role: importlib.import_module(name) for role, name in plan['entries'].items()}
        from src.next2 import deployment_training
        original_train = deployment_training.train_frozen_candidate
        def never_fit(*args, **kwargs):
            fit_calls.append('train_frozen_candidate')
            raise AssertionError('Empty input smoke must stop before fitting')
        deployment_training.train_frozen_candidate = never_fit
        for role, entry_name in plan['entries'].items():
            phase[0] = 'TRAIN_ADMISSION'
            entry = entries[role]
            output = workspace/'empty_input_outputs'/role
            try:
                entry.train(iter(()), output)
            except RuntimeError as error:
                if str(error) != 'Complete official training iterator required': raise
                admission = str(error)
            else:
                raise AssertionError('Empty official iterator must be rejected before fitting')
            assert not output.exists()
            class NoDatasets:
                def __iter__(self): raise AssertionError('Initial None must precede dataset access')
            phase[0] = 'INFERENCE'
            stream = entry.infer(NoDatasets(), payload/plan['models'][role])
            assert next(stream) is None
            stream.close()
            results[role] = {'entry_module': entry_name, 'empty_input_admission_error': admission,
                'empty_input_rejected_before_fit': True, 'training_output_directory_created': False,
                'ready_None_before_any_dataset_access': True, 'public_prediction_points': 0}
        deployment_training.train_frozen_candidate = original_train
        assert fit_calls == [] and denied == writes == network == []
        modules = {}
        for name, module in list(sys.modules.items()):
            if name != 'src' and not name.startswith(('src.', 'next2_', 'next3_')): continue
            source = getattr(module, '__file__', None)
            if source is None: continue
            path = Path(source).resolve(); assert path.is_relative_to(payload), name
            relative = path.relative_to(payload).as_posix()
            assert digest(path) == plan['files_sha256'][relative]
            modules[name] = {'relative': relative, 'sha256': digest(path)}
        for name, expected in plan['files_sha256'].items(): assert digest(payload/name) == expected
        assert modules
    finally:
        active[0] = False
    from src.next3.artifacts import atomic_json, utc_now
    report = {'status': 'PASS_ZERO_FIT_COPIED_INPUT_SMOKE', 'created_utc': utc_now(),
        'roles': results, 'copied_files_verified_before_and_after': len(plan['files_sha256']),
        'project_modules': modules, 'pinned_versions': versions, 'existing_environment': str(environment),
        'fresh_environment_claimed': False, 'raw_data_copied_or_read': False,
        'public_entry_bootstrap_before_training_module_imports': True,
        'new_fits': 0, 'fit_tripwire_calls': fit_calls, 'new_quality_scores': 0, 'seal_values_accessed': 0,
        'original_project_access_attempts': denied, 'unexpected_persistent_writes': writes, 'network_attempts': network,
        'input_manifest_sha256': digest(payload/'SMOKE_INPUT_MANIFEST.json'), 'elapsed_seconds': time.perf_counter()-began,
        'scope': 'Maintenance import/config/admission check from ordinary source copies using the existing pinned environment. Empty training input is intentionally rejected before fit; inference stops at ready None. Final fresh raw training and actual archive replay remain unrun.'}
    atomic_json(workspace/'RESULTS.json', report, immutable=True)
    print('NEXT3 ZERO-FIT COPIED INPUT SMOKE PASS TWO ROLES', flush=True)


if __name__ == '__main__': main()
