"""Check old champion bytes against the already published REAL archive manifest.

Read only archive metadata and its source manifest, never reduced predictions
or evaluation scalars. The old archive is retained unchanged.
"""
from pathlib import Path
import json
import subprocess
import sys
import zipfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.artifacts import sha256, write_json, utc_now


def main():
    desktop = Path(subprocess.check_output(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', "[Environment]::GetFolderPath('Desktop')"], text=True).strip())
    archive = desktop/'Structural_Break_GPT_전달용_REAL_20260908T160647Z.zip'
    expected_archive_sha = '3ab42b2f66a99f96c28c8cfdb382d4e076612599206d9b944e22c6057256e635'
    if sha256(archive) != expected_archive_sha:
        raise RuntimeError('The preceding published archive changed')
    with zipfile.ZipFile(archive) as handle:
        manifest_bytes = handle.read('SOURCE_MANIFEST.json')
    if manifest_bytes != (ROOT/'SOURCE_MANIFEST.json').read_bytes():
        raise RuntimeError('Prior published source manifest differs from the preserved local copy')
    manifest = json.loads(manifest_bytes)
    checks = {}
    for row in manifest['files']:
        relative = row['path']
        if relative.startswith(('artifacts/deployment/final/', 'artifacts/models/real/')) or relative in ('main.py', 'configs/deployment.json', 'DEV_CHAMPION_CANDIDATES.json'):
            if sha256(ROOT/relative) != row['sha256']:
                raise RuntimeError('A preserved prior champion artifact changed: '+relative)
            checks[relative] = row['sha256']
    if sum(path.endswith('model.joblib') for path in checks) != 3:
        raise RuntimeError('The three prior final champion models were not all checked')
    out = ROOT/'artifacts/next/prior_preservation'
    if (out/'RESULTS.json').exists():
        raise RuntimeError('Preserve the first prior-archive audit receipt')
    write_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(), 'old_archive': str(archive),
        'old_archive_sha256': expected_archive_sha, 'manifest_sha256': sha256(ROOT/'SOURCE_MANIFEST.json'),
        'preserved_files_sha256': checks, 'checked_prior_final_models': 3,
        'checked_prior_cv_models': sum(path.endswith('.joblib') and path.startswith('artifacts/models/real/') for path in checks),
        'script_sha256': sha256(Path(__file__)), 'archive_files_decoded': ['SOURCE_MANIFEST.json'],
        'reduced_evaluation_scalars_or_predictions_decoded': False, 'archive_modified': False})
    print('PRIOR CHAMPION PRESERVATION PASS', len(checks), flush=True)


if __name__ == '__main__':
    main()
