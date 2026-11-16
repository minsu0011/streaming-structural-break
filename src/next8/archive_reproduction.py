"""Execute the actual frozen inference from a newly extracted raw-free ZIP."""
from pathlib import Path
import datetime as dt
import os
import subprocess
import sys
from src.next8.common import ROOT, BASE, read_json, digest, atomic, utc_now, activity
from src.next8.delivery_payload import select_payload, materialize, zip_payload, verify_extract


def stamp_kst():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime('%Y%m%d_%H%M%S')


def validate_reproduction(record, fixture):
    assert record['status'] == 'PASS' and fixture['status'] == 'FROZEN'
    assert record['replay_new_process_fixture_bitwise'] and record['original_workspace_access_blocked']
    assert record['new_fits'] == record['new_quality_scores'] == record['competition_or_RP_cache_access'] == record['denied_read_attempts'] == 0
    assert record['runtime_manifest_sha256'] == fixture['runtime_manifest_sha256']
    assert record['implementation_lock_sha256'] == fixture['implementation_lock_sha256']
    assert record['synthetic_inputs'] == fixture['synthetic_inputs']
    assert len(record['checks']) == record['research_paths'] == 660
    assert len(record['canonical_public_entry_checks']) == record['canonical_paths'] == 9
    assert record['scalar_prediction_points'] == 669 * 1024
    for check in record['checks']:
        assert check['points'] == 1024
        assert check['fresh_scalar_chunk_bitwise'] and check['pickle_resume_at73_bitwise']
        assert check['future_suffix_prefix73_bitwise'] and check['only_H_and_current_scalar_given_to_state']
    for check in record['canonical_public_entry_checks']:
        assert check['points'] == 1024 and check['historical_NEXT7_fixture_bitwise_equal'] and check['public_infer_generator']
    assert len({(c['scope'], c['method'], c['case']) for c in record['checks']}) == 660
    assert record['checks'] == fixture['checks']
    assert record['canonical_public_entry_checks'] == fixture['canonical_public_entry_checks']
    return {'status': 'PASS', 'actual_research_paths': 660, 'actual_canonical_public_entry_paths': 9,
            'scalar_prediction_points': 669 * 1024, 'all_frozen_synthetic_vector_bits_equal': True,
            'scalar_chunk_pickle_restore_future_suffix_prefix_checks': 'PASS',
            'original_workspace_model_cache_and_competition_read_fallback_blocked': True,
            'new_fits': 0, 'new_quality_scores': 0, 'raw_data_retraining_performed': False}


def archive_and_reproduce(kind, final_directory=None):
    assert kind in ('CORE', 'FINAL')
    assert (kind == 'FINAL') == (final_directory is not None)
    parent = BASE / 'delivery'
    parent.mkdir(parents=True, exist_ok=True)
    assert not list(parent.glob(kind + '_*')), 'Previous partial delivery retained; use an explicit repair'
    token = stamp_kst()
    out = parent / (kind + '_' + token)
    out.mkdir(exist_ok=False)
    archive_name = ('STRUCTURAL_BREAK_NEXT8_VARIABLE_DYNAMICS_' + token + '.zip' if kind == 'FINAL'
                    else 'NEXT8_PORTABLE_CORE_' + token + '.zip')
    with activity(kind + ' portable archive selection and verified fresh extraction', 'FINAL_AUDIT', 'B7_NORMALIZATION_AND_LINEAGE'):
        paths = select_payload(final_directory)
        content = materialize(paths, out / 'payload')
        archive = out / archive_name
        archive_sha = zip_payload(out / 'payload', archive)
        extraction = out / 'fresh_extract'
        verification = verify_extract(archive, extraction)
        assert verification['archive_sha256'] == archive_sha
        atomic(out / 'ARCHIVE_INTEGRITY.json', {'timestamp': utc_now(), **verification,
               'files': len(content), 'archive_bytes': archive.stat().st_size})
    command = [str(ROOT / '.venv/Scripts/python.exe'), '-B',
               str(extraction / 'scripts/next8/reproduce_next8_portable.py'),
               '--output-directory', str(out / 'replay'), '--deny-original-root', str(ROOT)]
    env = dict(os.environ)
    env['PYTHONPATH'] = str(extraction)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    assert not env.get('NEXT5_TEST_AUDIT_DIRECTORY')
    with activity(kind + ' actual-model replay from freshly extracted ZIP', 'FINAL_AUDIT', 'B7_NORMALIZATION_AND_LINEAGE'):
        with (out / 'REPLAY.log').open('x', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=extraction, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            atomic(out / 'REPLAY_START.json', {'status': 'RUNNING', 'timestamp': utc_now(), 'pid': process.pid,
                   'parent_pid': os.getpid(), 'command': command, 'archive_sha256': archive_sha,
                   'activity_and_progress_not_a_fixed_elapsed_timeout': True})
            returncode = process.wait()
        assert returncode == 0, 'Fresh ZIP replay failure preserved in REPLAY.log'
        record = read_json(out / 'replay/REPRODUCTION_DONE.json')
        fixture = read_json(BASE / 'portable_fixture/FIXTURE.json')
        result = validate_reproduction(record, fixture)
        assert digest(out / 'replay/PREDICTIONS.npz') == record['predictions_sha256']
        # Runtime caches may be added; every delivered source/model/evidence
        # member must still have its original bytes after execution.
        for name, sha in content.items():
            assert digest(extraction / name) == sha, 'Payload member mutated during replay: ' + name
        receipt = {'timestamp': utc_now(), 'kind': kind, **result, 'archive_integrity': verification,
            'archive_path': archive.relative_to(ROOT).as_posix(), 'archive_sha256': archive_sha,
            'archive_bytes': archive.stat().st_size(), 'extracted_file_count': len(content),
            'payload_manifest_sha256': digest(extraction / 'PAYLOAD_CONTENT_MANIFEST_NEXT8.json'),
            'all_delivered_file_hashes_unchanged_after_replay': True,
            'reproduction_receipt_path': (out / 'replay/REPRODUCTION_DONE.json').relative_to(ROOT).as_posix(),
            'reproduction_receipt_sha256': digest(out / 'replay/REPRODUCTION_DONE.json'),
            'fixture_sha256': digest(BASE / 'portable_fixture/FIXTURE.json'),
            'command': command, 'log_sha256': digest(out / 'REPLAY.log'),
            'archive_self_hash_receipt_is_external_to_that_archive': True,
            'existing_environment_reused_without_package_installation': True}
        atomic(out / 'DONE.json', receipt)
    return receipt
