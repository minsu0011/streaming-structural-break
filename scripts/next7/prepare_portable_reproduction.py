"""Build an isolated source/model payload and deny original workspace reads."""
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.next7.common import BASE, read_json, save, digest, utc_now, activity, note
from src.next7.worker_environment import environment


def child(arguments, cwd, log):
    env = environment()
    env.pop('PYTHONPATH', None)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    with log.open('x', encoding='utf-8') as f:
        run = subprocess.run([sys.executable, '-B', *arguments], cwd=cwd, env=env,
            stdout=f, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    assert run.returncode == 0, 'Portable child failed; inspect ' + str(log)


def main(snapshot):
    assert snapshot and all(c.isalnum() or c in '_-' for c in snapshot)
    relative = f'reproduction/{snapshot}'
    directory = BASE / relative
    assert not directory.exists()
    directory.mkdir(parents=True)
    script = 'scripts/next7/reproduce_research_bundle.py'
    with activity('Raw-data-free source/model reproduction in isolated payload', 'REPRODUCTION_ENGINEERING', 'REPRODUCTION_AND_RUNTIME'):
        fixture = BASE / 'reproduction_fixture'
        if not fixture.exists():
            child([script, '--output-directory', str(fixture), '--create-fixture'], ROOT, directory / 'fixture.log')
        assert read_json(fixture / 'FIXTURE.json')['source_sha256'] == digest(ROOT / script)
        payload = directory / 'payload'
        payload.mkdir()
        files = [p for p in (ROOT / 'src').rglob('*.py') if '__pycache__' not in p.parts]
        files += [ROOT / script, ROOT / 'COUNTEREXAMPLE_SUITE_LOCK.json', ROOT / 'requirements.txt', ROOT / 'pyproject.toml']
        files += [ROOT / 'configs' / name for name in ('features.json', 'arch_input.json',
            'ar_residual_input.json', 'historical_context.json', 'historical_quantiles.json',
            'magnitude_dependence.json', 'rank_gaussian.json')]
        files += list((BASE / 'research_bundles').glob('fold_*/*'))
        files += list(fixture.iterdir())
        copied = {}
        for source in files:
            assert source.is_file() and not source.is_symlink()
            name = source.relative_to(ROOT)
            target = payload / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open('rb') as src, target.open('xb') as dst:
                shutil.copyfileobj(src, dst)
            assert digest(source) == digest(target)
            copied[name.as_posix()] = digest(target)
        save(relative + '/PAYLOAD_MANIFEST.json', {'status': 'FROZEN', 'created_utc': utc_now(),
            'files_sha256': copied, 'raw_data_included': False, 'original_large_caches_included': False,
            'nested_original_archives_included': False, 'models': 'Existing five research-only bundles, no refit'})
        output = payload / 'demo_output'
        child([script, '--output-directory', str(output), '--deny-original-root', str(ROOT)], payload,
            directory / 'isolated_reproduction.log')
        result = read_json(output / 'REPRODUCTION_DONE.json')
        assert result['status'] == 'PASS' and result['original_workspace_access_blocked']
        assert result['denied_access_attempts'] == 0 and len(result['checks']) == 90
        assert all(r['max_absolute_error'] <= 1e-12 for r in result['checks'])
        save(relative + '/DONE.json', {'status': 'PASS', 'created_utc': utc_now(),
            'payload_manifest_sha256': digest(directory / 'PAYLOAD_MANIFEST.json'),
            'reproduction_receipt_sha256': digest(output / 'REPRODUCTION_DONE.json'),
            'paths_checked': 90, 'scalar_outputs_checked': 92160,
            'all_bitwise_equal': all(r['bitwise_equal'] for r in result['checks']),
            'maximum_absolute_error': max(r['max_absolute_error'] for r in result['checks']),
            'original_source_data_artifact_access_blocked': True, 'fresh_subprocess': True,
            'new_training': False, 'source_sha256': digest(__file__)})
    note('Isolated raw-free research reproduction passed', 'Five fitted-fold bundles replay90 synthetic paths with original workspace reads denied', 'REPRODUCTION_AND_RUNTIME')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', required=True)
    args = parser.parse_args()
    main(args.snapshot)
