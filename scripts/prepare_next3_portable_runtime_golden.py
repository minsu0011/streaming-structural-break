"""Bind raw-free public-API golden outputs to frozen references before packaging."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[key] = '1'
from pathlib import Path
import importlib
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True
from src.next3.artifacts import BASE, read_json, atomic_json, digest, utc_now, progress
from src.next3.portable_probes import cases, replay, SEED
from scripts.prepare_next3_independent_reproduction import selected_performance


def main():
    out = BASE/'portable_runtime_golden'; out.mkdir(parents=True, exist_ok=False)
    selection, _ = selected_performance()
    directories = {'next2_shared_performance_main': 'artifacts/next2/canonical_deployment/PERFORMANCE_shared_v1',
        'next3_packed_performance_main': 'artifacts/next3/canonical_deployment/PERFORMANCE_packed_shared_v1',
        'next3_lean_performance_main': 'artifacts/next3/canonical_deployment/PERFORMANCE_lean_shared_v1'}
    references = {'DISTILLED': ('next2_canonical_distilled_main', 'artifacts/next2/canonical_deployment/DISTILLED_tree_used_v1'),
        'PERFORMANCE': ('next2_shared_performance_main', 'artifacts/next2/canonical_deployment/PERFORMANCE_shared_v1')}
    source_paths = ['src/next3/portable_probes.py', 'scripts/prepare_next3_portable_runtime_golden.py']
    atomic_json(out/'POLICY.json', {'created_utc': utc_now(), 'source_sha256': {p: digest(ROOT/p) for p in source_paths},
        'seed': SEED, 'selected_performance_entry': selection, 'references': references,
        'scope': 'Deterministic generated inputs for future package portability checks; no competition observations or quality score'}, immutable=True)
    values = cases(); records = {}
    for role, (module, directory) in references.items():
        entry = importlib.import_module(module); report = replay(entry, ROOT/directory, values)
        reversed_report = replay(entry, ROOT/directory, list(reversed(values)))
        forward = {r['case']: r for r in report['cases']}; reverse = {r['case']: r for r in reversed_report['cases']}
        assert forward == reverse
        if role == 'PERFORMANCE':
            selected = importlib.import_module(selection)
            actual = replay(selected, ROOT/directories[selection], values)
            assert actual == report
        records[role] = {**report, 'reference_entry': module, 'reference_model_sha256': digest(ROOT/directory/'model.sbnext2'),
            'reversed_case_order_exact': True, 'checked_entry_at_preparation': selection if role == 'PERFORMANCE' else module}
        progress('Raw-free public API golden preparation', role+' reference and fixed-order reversal exact', experiment=None)
    result = {'status': 'FROZEN_SYNTHETIC_PUBLIC_API_GOLDEN', 'created_utc': utc_now(),
        'policy_sha256': digest(out/'POLICY.json'), 'probe_source_sha256': digest(ROOT/'src/next3/portable_probes.py'),
        'roles': records, 'case_count': len(values), 'new_fits': 0, 'real_DEV_rows_accessed': 0, 'seal_rows': 0,
        'new_quality_scores': 0, 'raw_observations_included': False,
        'scope': 'A portability/determinism fixture; actual ZIP extraction and isolated archived-source replay remain required.'}
    atomic_json(out/'RESULTS.json', result, immutable=True)
    atomic_json(ROOT/'PORTABLE_RUNTIME_GOLDEN_NEXT3.json', result, immutable=True)
    print('NEXT3 PORTABLE GOLDEN PREPARED', len(values), sum(r['points'] for r in records.values()), flush=True)


if __name__ == '__main__': main()
