"""Compare old entry negative control with additive cold-import-safe entry."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.data import prepare_dev
from src.next.candidate_io import load_full_oof
from src.next.progress import checkpoint
from src.utils.artifacts import write_json, sha256, utc_now

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', default='V007_VARIANCE_FORECAST')
    parser.add_argument('--fixture-tag', default='v2')
    args = parser.parse_args()
    out = ROOT/'artifacts/next/cold_import_contract'/args.candidate
    out.mkdir(parents=True, exist_ok=True)
    if (out/'RESULTS.json').exists():
        raise RuntimeError('Keep completed cold-import evidence immutable')
    data = prepare_dev(ROOT)
    candidate, _, metadata, prediction, sources = load_full_oof(data, args.candidate)
    policy = json.loads((ROOT/'configs/next_latency_policy.json').read_text(encoding='utf-8'))
    selected = policy['series_ids'][:32]
    data.guard.admit(selected, purpose='cold-import guarded API fixture')
    fixture_path = ROOT/'data/next/cold_import_contract'/args.candidate/'fixture.json'
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    series = []
    for sid in selected:
        historical, online, _ = data.get(sid)
        rows = metadata['dataset_id'] == sid
        if set(metadata['fold'][rows]) != {0}:
            raise RuntimeError('Cold-import fixture must use fold-zero held-out series')
        series.append({'historical': historical.tolist(), 'online': online.tolist(), 'expected': prediction[rows].tolist()})
    model_directory = ROOT/'artifacts/next/deployment/contracts'/args.candidate/args.fixture_tag
    write_json(fixture_path, {'model_directory': str(model_directory), 'series': series})
    files = ['next_readonly_main.py', 'src/next/readonly_bootstrap.py', 'scripts/check_next_cold_import_child.py', 'scripts/verify_next_cold_import.py']
    identity = {'candidate': args.candidate, 'fixture_sha256': sha256(fixture_path), 'oof_sources': sources,
        'source_sha256': {path: sha256(ROOT/path) for path in files}, 'created_utc': utc_now(),
        'fixture_values_excluded_from_handoff': True, 'seal_rows': 0, 'reduced_new_uses': 0}
    write_json(out/'POLICY.json', identity)
    for entry, negative in [('next_main', True), ('next_readonly_main', False)]:
        output = out/(entry+'.json')
        command = [sys.executable, str(ROOT/'scripts/check_next_cold_import_child.py'), '--entry', entry,
            '--fixture', str(fixture_path), '--output', str(output)]
        if negative:
            command.append('--expect-import-write')
        subprocess.run(command, cwd=ROOT, check=True)
    positive = json.loads((out/'next_readonly_main.json').read_text(encoding='utf-8'))
    if positive['status'] != 'PASS' or positive['audit']['filesystem_mutation_attempts']:
        raise RuntimeError('Cold-import-safe entry failed')
    write_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(), 'candidate': args.candidate,
        'entrypoint': 'next_readonly_main.py', 'cold_import_and_first_jit_and_inference_write_denial': True,
        'old_next_entry_import_write_reproduced_and_prevented': True, 'points': positive['points'], 'series': positive['series'],
        'files_sha256': {name: sha256(out/name) for name in ['next_main.json', 'next_readonly_main.json']},
        'policy_sha256': sha256(out/'POLICY.json'), 'seal_rows': 0, 'reduced_new_uses': 0,
        'scope': 'Fresh process with Python audit-hook global write denial beginning before project/NumPy/Numba imports; preserves the previously verified infer math. Native-library syscall interception or cloud execution is not claimed.'})
    checkpoint(ROOT, 'cold import safe NEXT entry', ['negative-control import-write reproduction', 'fresh-process read-only import and exact OOF inference'],
        ['finalist engineering', 'remaining research backlog'], artifacts=[str((out/'RESULTS.json').relative_to(ROOT))], best=.611292686371357)
    print('COLD IMPORT CONTRACT COMPLETE', args.candidate, flush=True)
