"""Final raw-byte, protected-file, model, environment and membership audit."""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.metadata
import io
import re
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next5.artifacts import BASE, atomic_json, read_json, digest, utc_now, progress
from src.next5.guard import ResearchGuard, ShadowAccessError


def verify_folder(folder, records):
    folder = Path(folder)
    assert len(records) == 20
    assert {p.name for p in folder.iterdir()} == set(records)
    for name, row in records.items():
        path = folder / name
        assert path.is_file() and path.stat().st_size == row['bytes']
        assert digest(path) == row['sha256']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    if not re.fullmatch('[A-Za-z0-9_]+', args.tag):
        raise ValueError('Safe immutable final-audit tag required')
    session = read_json(BASE / 'SESSION.json')
    assert session.get('audit_started_utc')
    out = BASE / 'final_audit' / args.tag / 'frozen_boundary'
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / 'START.json', {'created_utc': utc_now(), 'source_sha256': digest(Path(__file__)),
        'audit_started_utc': session['audit_started_utc']}, immutable=True)
    refs = read_json(BASE / 'preflight/FROZEN_REFERENCES.json')
    context = read_json(BASE / 'preflight/INITIAL_CONTEXT.json')
    snapshot_path = BASE / 'preflight/PREEXISTING_SOURCE_AND_REPORT_SNAPSHOT.json'
    assert digest(snapshot_path) == context['snapshot_sha256']
    snapshot = read_json(snapshot_path)
    assert len(snapshot['files']) == context['preexisting_files'] == 1360
    guard = ResearchGuard()
    assert guard.shadow_lock_sha == refs['shadow_lock_sha256']
    assert guard.split.digest == refs['research_split_sha256']
    assert guard.lock_sha == refs['seal_lock_sha256']
    assert guard.ledger_sha == refs['reduced_ledger_sha256']
    assert (len(guard.research_ids), len(guard.shadow_ids), len(guard.seal_ids)) == (6400, 1600, 2000)
    blocked = {}
    for purpose in ('training', 'validation', 'feature cache', 'OOF', 'candidate selection', 'bootstrap', 'label access'):
        for kind, ids in [('Shadow', guard.shadow_ids), ('LOCAL_SEAL', guard.seal_ids)]:
            try:
                guard.admit([min(ids)], purpose=purpose)
            except ShadowAccessError:
                blocked[kind + ':' + purpose] = True
            else:
                raise AssertionError('Forbidden identifier admitted')
    progress('Final frozen data boundary', 'Whole-file byte hashes only; protected observations are not decoded',
             phase='FINAL_AUDIT', counts_as_research=False)
    raw = {name: digest(ROOT / 'data/raw' / name) for name in refs['raw_sha256']}
    assert raw == refs['raw_sha256']
    ledger_original = (BASE / 'preflight/retained/EXPERIMENT_LEDGER.csv').read_bytes()
    assert hashlib.sha256(ledger_original).hexdigest() == snapshot['files']['EXPERIMENT_LEDGER.csv']['sha256']
    ledger_now = (ROOT / 'EXPERIMENT_LEDGER.csv').read_bytes()
    assert ledger_now.startswith(ledger_original), 'Original ledger bytes must be an unchanged prefix'
    old_rows = list(csv.DictReader(io.StringIO(ledger_original.decode('utf-8-sig'))))
    all_rows = list(csv.DictReader(io.StringIO(ledger_now.decode('utf-8-sig'))))
    assert len(old_rows) == 276 and all_rows[:len(old_rows)] == old_rows
    assert all(row['ledger_section'] == 'NEXT5' for row in all_rows[len(old_rows):])
    for name, row in snapshot['files'].items():
        if name == 'EXPERIMENT_LEDGER.csv':
            continue
        assert digest(ROOT / name) == row['sha256'], name
    for name, sha in context['prior_session_sha256'].items():
        assert digest(ROOT / name) == sha
    assert read_json(ROOT / 'artifacts/next4/SESSION.json')['status'] == 'COMPLETE'
    old3 = read_json(ROOT / 'artifacts/next3/SESSION.json')
    assert old3['status'] == 'STOPPED_BY_USER' and old3['automatic_resume_authorized'] is False
    saved_patch = (BASE / 'preflight/INITIAL_STAGED_PATCH.diff').read_bytes()
    patch = subprocess.check_output(['git', 'diff', '--cached', '--binary'], cwd=ROOT, stderr=subprocess.PIPE)
    assert patch == saved_patch and hashlib.sha256(patch).hexdigest() == context['staged_patch_sha256']
    assert digest(ROOT / 'docs/RESEARCH_PROTOCOL_NEXT5.md') == context['specification_sha256']
    model_hashes = {}
    for name, role in refs['roles'].items():
        path = ROOT / role['model_file']
        assert digest(path) == role['model_sha256']
        model_hashes[name] = role['model_sha256']
    pins = {}
    for line in (ROOT / 'requirements-lock.txt').read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        name, expected = line.split('==')
        actual = importlib.metadata.version(name)
        assert actual == expected
        pins[name] = actual
    assert pins == refs['exact_environment_pins'] and len(pins) == 63
    assert sys.version == refs['python']
    progress('Final prior-work preservation audit', '1360 original files, staged bytes, frozen model weights and 63 exact environment pins checked',
             phase='FINAL_AUDIT', counts_as_research=False)
    pointer = read_json(ROOT / 'artifacts/next4/handoff/DESKTOP_DELIVERY_POINTER.json')
    assert digest(ROOT / pointer['receipt']) == pointer['sha256']
    old4_delivery = read_json(ROOT / pointer['receipt'])
    verify_folder(old4_delivery['twenty_file_folder'], old4_delivery['twenty_files'])
    archives = {}
    for key in ('full_project_archive', 'twenty_file_archive'):
        row = old4_delivery[key]
        path = Path(row['path'])
        assert path.stat().st_size == row['bytes'] and digest(path) == row['sha256']
        archives[str(path)] = row['sha256']
    old3_path = ROOT / 'artifacts/next3/handoff/user_stop_20260910T112445Z/DELIVERY_VERIFICATION.json'
    old3_delivery = read_json(old3_path)
    verify_folder(old3_delivery['folder'], old3_delivery['files'])
    archive3 = Path(old3_delivery['archive'])
    assert archive3.stat().st_size == old3_delivery['archive_bytes']
    assert digest(archive3) == old3_delivery['archive_sha256']
    archives[str(archive3)] = old3_delivery['archive_sha256']
    shadow = read_json(BASE / 'shadow/STATE.json')
    assert shadow['status'] == 'CLOSED' and shadow['shadow_opened'] is False
    assert shadow['labels_loaded'] is False and shadow['evaluations_completed'] == 0
    assert session['new_reduced_uses'] == session['new_public_submissions'] == session['new_cloud_submissions'] == 0
    guard.check_state()
    atomic_json(out / 'RESULTS.json', {
        'created_utc': utc_now(), 'status': 'PASS', 'audit_started_utc': session['audit_started_utc'],
        'raw_byte_sha256': raw, 'raw_observations_decoded_by_this_check': False,
        'research_series': 6400, 'shadow_series': 1600, 'LOCAL_SEAL_series': 2000,
        'all_seals_unopened': True, 'Shadow_status': shadow, 'blocked_admissions': blocked,
        'application_guard_not_OS_sandbox': True,
        'protected_source_and_report_files': 1360, 'original_ledger_rows': 276,
        'original_ledger_bytes_preserved': True, 'NEXT5_ledger_rows_appended': len(all_rows) - len(old_rows),
        'staged_patch_byte_equal': True, 'staged_patch_sha256': context['staged_patch_sha256'],
        'prior_session_states_byte_equal': True, 'old_Desktop_folders_verified': 2,
        'old_Desktop_folder_files_verified': 40, 'old_Desktop_archive_sha256': archives,
        'frozen_model_sha256': model_hashes, 'exact_environment_pins': pins,
        'new_reduced_uses': 0, 'new_public_or_cloud_submission_by_this_check': 0,
        'external_account_activity_audited': False, 'new_fits': 0,
        'source_sha256': digest(Path(__file__)), 'preexisting_snapshot_sha256': digest(snapshot_path),
    }, immutable=True)
    progress('Final frozen boundary and protected work verified', 'Original models, raw bytes, all old source/index work and both prior Desktop deliveries preserved',
             phase='FINAL_AUDIT', counts_as_research=False)


if __name__ == '__main__':
    main()
