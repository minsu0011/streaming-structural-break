"""Verified final GPT ZIP and exactly twenty Desktop process/result files."""
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next5.artifacts import BASE, atomic_json, read_json, digest, utc_now, progress
from src.next5.finalization import require_handoff_time
from src.next5.guard import ResearchGuard
from src.next5.handoff_payload import collect
from scripts.package_next4_handoff import write_verified_archive

TWENTY = '''00_GPT_READ_FIRST_NEXT5.md 01_GPT_HANDOFF_NEXT5.md MODEL_RESULTS_NEXT5.csv
CV_RESULTS_NEXT5.csv PAIRED_BOOTSTRAP_NEXT5.csv ALT_RESULTS_NEXT5.csv REGIME_RESULTS_NEXT5.csv
MECHANISM_RESULTS_NEXT5.csv TIME_FRACTURE_NEXT5.csv BREAK_AGE_NEXT5.csv
COUNTEREXAMPLE_RESULTS_NEXT5.csv LONG_NULL_RESULTS_NEXT5.csv OUTLIER_IMPULSE_NEXT5.csv
LATENCY_NEXT5.json LEAKAGE_AUDIT_NEXT5.json REPRODUCTION_NEXT5.json FINAL_AUDIT_NEXT5.json
BOTTLENECK_HISTORY_NEXT5.csv LIMITATIONS_NEXT5.md NEXT_SESSION_PLAN_NEXT5.md'''.split()
assert len(TWENTY) == len(set(TWENTY)) == 20


def main():
    timing = require_handoff_time()
    guard = ResearchGuard()
    guard.check_state()
    audit = read_json(ROOT / 'FINAL_AUDIT_NEXT5.json')
    selection = read_json(ROOT / 'DEV_CANDIDATES_NEXT5.json')
    assert audit['status'] == 'PASS' and audit['checks'] and all(v is True for v in audit['checks'].values())
    assert selection['status'] == 'FROZEN_NEXT5_FINAL_ROLES'
    for name, sha in audit['evidence_sha256'].items():
        assert digest(ROOT / name) == sha
    progress('Verified NEXT5 Desktop packaging', 'Observed research and separate audit minimums passed', phase='DELIVERY')
    paths, files, excluded, missing = collect(require_final=True)
    assert not missing and all(name in files for name in TWENTY)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = BASE / 'handoff' / stamp
    out.mkdir(parents=True, exist_ok=False)
    manifest = {'created_utc': utc_now(), 'status': 'FROZEN_NEXT5_PACKAGE_SOURCE_MANIFEST',
        'files': files, 'files_excluding_this_manifest': len(files),
        'uncompressed_bytes': sum(r['bytes'] for r in files.values()),
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'final_audit_sha256': digest(ROOT / 'FINAL_AUDIT_NEXT5.json'),
        'selection_sha256': digest(ROOT / 'DEV_CANDIDATES_NEXT5.json'),
        'observed_time_before_packaging': timing, 'excluded_reports': excluded,
        'exclusions': ['Raw competition observations and labels', 'LOCAL_SEAL values and labels',
            'Full real point OOF and feature caches', 'Credentials, environments, Git and old archives',
            'Nested copied payloads and live supervisor state'],
        'fitted_models': 'Existing frozen roles and research artifacts; included research models are not promoted roles.',
        'small_arrays': 'Fixed bootstrap multiplicities or aggregate score draws only; no real point predictions.',
        'portable_use': 'Cold inference replay needs no competition raw data; retraining requires separately supplied hash-matched raw files and excluded RP caches regenerated under the guard.',
        'self_hash': 'Manifest and complete archive are hashed in the external Desktop verification receipt.',
        'source_sha256': digest(Path(__file__)),
        'archive_writer_source_sha256': digest(ROOT / 'scripts/package_next4_handoff.py')}
    atomic_json(out / 'SOURCE_MANIFEST_NEXT5.json', manifest, immutable=True)
    atomic_json(ROOT / 'SOURCE_MANIFEST_NEXT5.json', manifest, immutable=True)
    manifest_path = ROOT / 'SOURCE_MANIFEST_NEXT5.json'
    members = [(p, p.relative_to(ROOT).as_posix(), files[p.relative_to(ROOT).as_posix()]['sha256']) for p in paths]
    members.append((manifest_path, manifest_path.name, digest(manifest_path)))
    desktop = Path(subprocess.check_output(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command',
        "[Environment]::GetFolderPath('Desktop')"], text=True).strip()).resolve()
    assert desktop.is_dir()
    archive_path = desktop / ('STRUCTURAL_BREAK_GPT_HANDOFF_NEXT5_' + stamp + '.zip')
    assert archive_path.resolve().parent == desktop
    full = write_verified_archive(archive_path, members)
    folder = desktop / ('Structural_Break_NEXT5_과정과결과_20개_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    assert folder.resolve().parent == desktop
    folder.mkdir(exist_ok=False)
    twenty = {}
    for name in TWENTY:
        destination = folder / name
        shutil.copyfile(ROOT / name, destination)
        assert digest(destination) == files[name]['sha256']
        twenty[name] = {'sha256': digest(destination), 'bytes': destination.stat().st_size}
    assert len(list(folder.iterdir())) == 20 and all(p.is_file() for p in folder.iterdir())
    summary_archive = write_verified_archive(folder.with_suffix('.zip'), [(folder / n, n, twenty[n]['sha256']) for n in TWENTY])
    guard.check_state()
    receipt = {'created_utc': utc_now(), 'status': 'PASS', 'full_project_archive': full,
        'twenty_file_folder': str(folder), 'twenty_file_count': 20, 'twenty_files': twenty,
        'twenty_file_archive': summary_archive, 'manifest_sha256': digest(manifest_path),
        'raw_competition_data_included': False, 'real_point_OOF_included': False, 'credentials_included': False,
        'shadow_opened': False, 'all_seals_unopened': True, 'new_reduced_usage': 0,
        'new_public_submission': 0, 'new_cloud_submission': 0,
        'observed_time_before_packaging': timing, 'observed_time_after_packaging': require_handoff_time(),
        'source_sha256': digest(Path(__file__))}
    receipt_path = out / 'DESKTOP_DELIVERY_VERIFICATION_NEXT5.json'
    atomic_json(receipt_path, receipt, immutable=True)
    atomic_json(BASE / 'handoff/DESKTOP_DELIVERY_POINTER.json', {'created_utc': utc_now(),
        'receipt': receipt_path.relative_to(ROOT).as_posix(), 'sha256': digest(receipt_path)}, immutable=True)
    print('NEXT5 DESKTOP DELIVERY VERIFIED', full['entries'], 'full-project files; exactly 20 summary files', flush=True)
    print('NEXT5 FULL ZIP', archive_path.name, full['sha256'], flush=True)


if __name__ == '__main__':
    main()
