"""Create exactly 20 review files only after the actual Desktop ZIP replay passes."""
from pathlib import Path
import datetime
import shutil
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, atomic_json, digest, progress, read_json, utc_now
from src.next3.final_validation import final_clock, successful_receipt

ROOT_FILES = ('00_GPT_READ_FIRST_NEXT3.md', '01_GPT_HANDOFF_NEXT3.md', 'BOTTLENECK_BOARD_FINAL.json',
    'MODEL_RESULTS_NEXT3.csv', 'CV_RESULTS_NEXT3.csv', 'PAIRED_BOOTSTRAP_NEXT3.csv',
    'ALT_SPLIT_RESULTS_NEXT3.csv', 'REGIME_RESULTS_NEXT3.csv', 'MECHANISM_RESULTS_NEXT3.csv',
    'COUNTEREXAMPLE_RESULTS_NEXT3.csv', 'LONG_NULL_RESULTS_NEXT3.csv', 'LATENCY_NEXT3.json',
    'STATE_MEMORY_NEXT3.json', 'DEV_CANDIDATES_NEXT3.json', 'FINAL_AUDIT_NEXT3.json',
    'REPRODUCTION_NEXT3.json', 'DETERMINISM_NEXT3.json', 'LIMITATIONS_NEXT3.md', 'NEXT_SESSION_PLAN_NEXT3.md')


def main():
    timing = final_clock(read_json(BASE/'SESSION.json'), datetime.datetime.now(datetime.timezone.utc),
                         require_complete_duration=True)
    verification = BASE/'handoff/DESKTOP_ZIP_VERIFICATION_NEXT3.json'
    receipt = successful_receipt(verification)
    assert receipt['actual_archive_replay'] == 'PASS'
    assert receipt.get('documented_reviewer_command') == 'PASS'
    reviewer_path = BASE/'handoff/reviewer_command/RESULTS.json'
    successful_receipt(reviewer_path)
    assert digest(reviewer_path) == receipt['documented_reviewer_command_sha256']
    archive = Path(receipt['archive']); assert digest(archive) == receipt['archive_sha256']
    assert digest(BASE/'handoff/actual_archive_replay/RESULTS.json') == receipt['actual_archive_replay_sha256']
    successful_receipt(BASE/'handoff/actual_archive_replay/RESULTS.json')
    audit = successful_receipt(ROOT/'FINAL_AUDIT_NEXT3.json')
    assert audit['checks'] and all(type(value) is bool and value for value in audit['checks'].values())
    manifest_path = ROOT/'SOURCE_MANIFEST_NEXT3.json'
    assert digest(manifest_path) == receipt['manifest_sha256']
    manifest = read_json(manifest_path)
    assert len(ROOT_FILES) == 19 and len(set(ROOT_FILES)) == 19
    sources = [ROOT/name for name in ROOT_FILES]+[verification]
    for path in sources:
        assert path.is_file() and not path.is_symlink() and not path.is_junction()
        if path != verification:
            assert digest(path) == manifest['files'][path.name]['sha256'], 'Root review file differs from actual archive'
    desktop = Path(subprocess.check_output(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command',
        "[Environment]::GetFolderPath('Desktop')"], text=True).strip()).resolve()
    assert desktop.is_dir()
    stamp = datetime.datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f')
    folder = desktop/('Structural_Break_NEXT3_핵심결과_20개_'+stamp)
    assert folder.parent == desktop and not folder.exists()
    folder.mkdir(); files = {}
    for path in sources:
        target = folder/path.name; shutil.copyfile(path, target)
        sha = digest(path); assert digest(target) == sha
        files[target.name] = {'sha256': sha, 'bytes': target.stat().st_size, 'source': str(path)}
    assert len(files) == len(list(folder.iterdir())) == 20
    assert all(path.is_file() and not path.is_symlink() and not path.is_junction() for path in folder.iterdir())
    result = {'status': 'PASS', 'created_utc': utc_now(), 'directory': str(folder), 'file_count': 20,
        'files': files, 'archive': str(archive), 'archive_sha256': receipt['archive_sha256'],
        'archive_replay_passed': True, '19_root_reports_equal_actual_archive_members': True,
        '20th_file': 'External actual Desktop ZIP and replay verification receipt',
        'original_project_files_moved': False, 'old_desktop_folders_modified': False, 'clock': timing,
        'script_sha256': digest(Path(__file__))}
    atomic_json(BASE/'handoff/DESKTOP_TOP20_VERIFICATION_NEXT3.json', result, immutable=True)
    progress('NEXT3 Desktop top 20 verified', 'Exactly 20 copied review files, with 19 matching actual archive members and external verified ZIP/replay receipt', phase='FINAL_AUDIT')
    print('NEXT3 DESKTOP TOP20 PASS', folder, flush=True)


if __name__ == '__main__': main()
