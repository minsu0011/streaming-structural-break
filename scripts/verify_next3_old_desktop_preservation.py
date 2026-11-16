"""Read-only byte verification of the user's earlier Desktop ZIP and 20 files."""
from pathlib import Path
import argparse
import datetime
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, atomic_json, digest, progress, read_json, utc_now

OLD_ZIP_SHA256 = '46ace7564b215a44776fa467f5b55de8cbe614329e8622430751661c22997f40'
OLD_NAMES = ('00_GPT_READ_FIRST_NEXT2.md', '01_GPT_HANDOFF_NEXT2.md', 'ALT_SPLIT_RESULTS_NEXT2.csv',
    'COMPLEMENT_BLEND_RESULTS.csv', 'COUNTEREXAMPLE_RESULTS.csv', 'CURRENT_GENERATION_COMPONENT_GRAPH.md',
    'CV_RESULTS_NEXT2.csv', 'DEV_CANDIDATES_NEXT2.json', 'DISTILLATION_RESULTS.csv', 'EXPERIMENT_LEDGER.csv',
    'FINAL_AUDIT_NEXT2.json', 'LIMITATIONS_NEXT2.md', 'LONG_NULL_RESULTS_NEXT2.csv', 'MECHANISM_RESULTS_NEXT2.csv',
    'MODEL_RESULTS_NEXT2.csv', 'NEXT_SESSION_PLAN_NEXT2.md', 'PAIRED_BOOTSTRAP_NEXT2.csv', 'REGIME_RESULTS_NEXT2.csv',
    'REPRODUCTION_NEXT2.json', 'RUNTIME_PRUNING_RESULTS.csv')


def verify_existing_desktop_deliveries(root):
    root = Path(root).resolve(); base = root/'artifacts/next3'
    frozen_path = base/'preflight/FROZEN_REFERENCES.json'; frozen = read_json(frozen_path)
    previous_path = root/'artifacts/next2/handoff/DESKTOP_ZIP_VERIFICATION_NEXT2.json'
    previous = read_json(previous_path); archive = Path(previous['archive'])
    assert previous['status'] == 'PASS' and previous['sha256'] == OLD_ZIP_SHA256
    assert datetime.datetime.fromisoformat(previous['created_utc']) < datetime.datetime.fromisoformat(read_json(base/'SESSION.json')['started_utc'])
    if not archive.is_file() or archive.is_symlink() or archive.is_junction():
        raise RuntimeError('Earlier Desktop archive must remain an ordinary file')
    if digest(archive) != OLD_ZIP_SHA256 or archive.stat().st_size != previous['bytes']:
        raise RuntimeError('Earlier Desktop archive bytes changed')
    folders = list(archive.parent.glob('Structural_Break_*_20260910_083240_756'))
    if len(folders) != 1 or not folders[0].is_dir(): raise RuntimeError('Expected earlier Desktop 20-file folder missing or ambiguous')
    folder = folders[0]
    if folder.is_symlink() or folder.is_junction(): raise RuntimeError('Earlier Desktop folder must remain ordinary')
    if {path.name for path in folder.iterdir()} != set(OLD_NAMES):
        raise RuntimeError('Earlier Desktop 20-file membership changed')
    files = {}
    for name in OLD_NAMES:
        path = folder/name
        if not path.is_file() or path.is_symlink() or path.is_junction():
            raise RuntimeError('Earlier Desktop review file is missing or linked: '+name)
        expected = frozen['protected_sha256'][name]
        actual = digest(path)
        if actual != expected: raise RuntimeError('Earlier Desktop file differs from initial frozen project bytes: '+name)
        files[name] = {'sha256':actual, 'bytes':path.stat().st_size, 'initial_project_source':name}
    return {'status':'PASS', 'created_utc':utc_now(), 'archive':str(archive), 'archive_sha256':OLD_ZIP_SHA256,
        'archive_bytes':archive.stat().st_size, 'directory':str(folder), 'file_count':len(files), 'files':files,
        'all_old_Desktop_delivery_bytes_unchanged':True, 'all_old_20_files_match_initial_project_frozen_hashes':True,
        'initial_frozen_references_sha256':digest(frozen_path), 'previous_archive_receipt_sha256':digest(previous_path),
        'read_only_verification':True, 'old_files_moved_or_rewritten':False}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--label', default='maintenance_v1'); args = parser.parse_args()
    if not args.label.isidentifier(): raise ValueError('Unique simple preserved-result label required')
    result = verify_existing_desktop_deliveries(ROOT)
    out = BASE/'maintenance'/('old_desktop_preservation_'+args.label)
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out/'RESULTS.json', {**result, 'source_sha256':digest(Path(__file__))}, immutable=True)
    progress('Previous Desktop handoff preserved','Earlier NEXT2 ZIP SHA and all 20 existing result-file bytes agree with their previous authorities; read-only check')
    print('NEXT3 OLD DESKTOP PRESERVATION PASS',result['file_count'],'FILES AND ORIGINAL ZIP',flush=True)


if __name__ == '__main__': main()
