"""Receipt plumbing integration only; synthetic fixture checks are simulated."""
import copy
import hashlib
import json
import zipfile
from contextlib import nullcontext
from pathlib import Path

from src.next8.common import atomic, digest
from src.next8 import archive_reproduction_v3 as archive
from src.next8 import delivery_payload as payload
from src.next8 import desktop_delivery_v2 as desktop


def test_archive_writes_receipt_after_verified_zip_and_simulated_replay(tmp_path, monkeypatch):
    root = tmp_path / 'workspace'
    base = root / 'artifacts/next8/run_test'
    base.mkdir(parents=True)
    source = root / 'model.txt'
    source.write_text('test-only model asset', encoding='utf-8')
    checks = [dict(scope=str(i), method='M0', case='toy', points=1024,
                   fresh_scalar_chunk_bitwise=True, pickle_resume_at73_bitwise=True,
                   future_suffix_prefix73_bitwise=True, only_H_and_current_scalar_given_to_state=True)
              for i in range(660)]
    canonical = [dict(points=1024, historical_NEXT7_fixture_bitwise_equal=True,
                      public_infer_generator=True) for _ in range(9)]
    fixture = dict(status='FROZEN', runtime_manifest_sha256='a', implementation_lock_sha256='b',
                   synthetic_inputs={}, checks=checks, canonical_public_entry_checks=canonical)
    atomic(base / 'portable_fixture/FIXTURE.json', fixture)
    atomic(base / 'PORTABLE_CACHE_PATH_IMPLEMENTATION_LOCK_NEXT8.json', {})
    record = dict(copy.deepcopy(fixture), status='PASS', replay_new_process_fixture_bitwise=True,
                  original_workspace_access_blocked=True, new_fits=0, new_quality_scores=0,
                  competition_or_RP_cache_access=0, denied_read_attempts=0, research_paths=660,
                  canonical_paths=9, scalar_prediction_points=669 * 1024,
                  fresh_compiler_cache_relative_to_payload='nc',
                  original_read_boundary_and_numerical_sources_unchanged=True,
                  portable_cache_path_implementation_lock_sha256=digest(base / 'PORTABLE_CACHE_PATH_IMPLEMENTATION_LOCK_NEXT8.json'))

    class SimulatedReplay:
        pid = 12345

        def __init__(self, command, **kwargs):
            output = Path(command[command.index('--output-directory') + 1])
            output.mkdir()
            vector = output / 'PREDICTIONS.npz'
            vector.write_bytes(b'test-only simulated predictions')
            atomic(output / 'REPRODUCTION_DONE.json', dict(record, predictions_sha256=digest(vector)))
            kwargs['stdout'].write('Simulated replay: tests receipt plumbing only\n')

        def wait(self):
            return 0

    monkeypatch.setattr(archive, 'ROOT', root)
    monkeypatch.setattr(archive, 'BASE', base)
    monkeypatch.setattr(payload, 'BASE', base)
    monkeypatch.setattr(archive, 'select_payload', lambda _: {'model.txt': source})
    monkeypatch.setattr(archive, 'activity', lambda *args: nullcontext())
    monkeypatch.setattr(archive.subprocess, 'Popen', SimulatedReplay)
    result = archive.archive_and_reproduce('CORE')
    assert result['archive_bytes'] == (root / result['archive_path']).stat().st_size
    assert result['archive_bytes'] > 0
    assert result['status'] == 'PASS'
    assert result['all_delivered_file_hashes_unchanged_after_replay']
    assert json.loads((root / result['archive_path']).with_name('DONE.json').read_text()) == result


def test_twenty_file_delivery_writes_final_receipt_and_preserves_prior_files(tmp_path, monkeypatch):
    root = tmp_path / 'workspace'
    base = root / 'artifacts/next8/run_test'
    final = root / 'final'
    desk = tmp_path / 'desktop'
    for p in (base, final, desk):
        p.mkdir(parents=True)
    prior = desk / 'prior'
    prior.mkdir()
    (prior / 'keep.txt').write_text('unchanged', encoding='utf-8')
    atomic(base / 'RESTORED_NEXT7_AUTHORITY.json', dict(previous_Desktop_directory=str(prior),
           previous_Desktop_files_sha256={'keep.txt': digest(prior / 'keep.txt')}))
    policy_names = ('NEXT8_PROTOCOL_LOCK.json', 'NEXT8_FEATURE_BUDGET.json', 'NEXT8_MODEL_MENU_LOCK.json',
                    'FINAL_MODEL_IMPLEMENTATION_POLICY_NEXT8.json', 'CERTIFICATION_AND_SCORING_POLICY_NEXT8.json',
                    'SELF_HISTORY_CENTER_POLICY_NEXT8.json', 'NEXT8_PROTOCOL_SUPPLEMENT_H_CONTEXT.json')
    for name in policy_names:
        atomic(base / name, {'test_only': True})
    for name in ('FIT_ACCOUNTING_NEXT8.json', 'TIME_ACCOUNTING_NEXT8.json', 'SOURCE_MANIFEST_NEXT8.json',
                 'FINAL_AUDIT_NEXT8.json', 'LEAKAGE_AUDIT_NEXT8.json', 'NEXT8_DATA_ACCESS_AUDIT.json',
                 'LATENCY_NEXT8.json', 'DETERMINISM_NEXT8.json', 'REPRODUCTION_NEXT8.json',
                 'VARIABLE_CERTIFICATION_REGISTRY.json', 'PREDICTIVE_FACTOR_GRAPH.json'):
        atomic(final / name, {'test_only': True})
    for name in ('00_READ_FIRST_NEXT8_KO.md', 'RESEARCH_REPORT_NEXT8.md', 'FEATURE_EXPLANATION_CARDS_KO.md',
                 'NEXT_SESSION_PLAN_NEXT8.md', 'SELF_HISTORY_BASELINE_LIMIT_KO.md'):
        atomic(final / name, 'Test-only document\n')
    for name in ('FEATURE_CLASSIFICATION_NEXT8.csv', 'TARGET_DIRECTION_ATLAS.csv', 'CONDITIONAL_DIRECTION_RESULTS.csv',
                 'MOVEMENT_FORECAST_RESULTS.csv', 'MODEL_COMPARISON_NEXT8.csv', 'PAIRED_UNCERTAINTY_NEXT8.csv',
                 'EXPERIMENT_LEDGER_NEXT8.csv', 'MOVEMENT_FACTOR_ABLATIONS.csv', 'SELF_HISTORY_CENTER_CONTROL_NEXT8.csv',
                 'TIME_GENERALIZATION_NEXT8.csv', 'H_BACKGROUND_RESULTS_NEXT8.csv', 'COUNTEREXAMPLE_RESULTS_NEXT8.csv',
                 'NEGATIVE_CONTROLS_NEXT8.csv'):
        atomic(final / name, 'test_only\n1\n')
    atomic(base / 'FIGURES/NEXT8_EIGHT_FIGURES.pdf', 'Test-only figure bytes')
    full = root / 'test_full.zip'
    with zipfile.ZipFile(full, 'x') as stream:
        stream.writestr('test_only.txt', 'test-only payload')
    atomic(root / 'test_replay.json', {'status': 'PASS', 'test_only': True})
    reproduction = dict(status='PASS', kind='FINAL', all_frozen_synthetic_vector_bits_equal=True,
                        archive_path=full.name, archive_sha256=digest(full),
                        reproduction_receipt_path='test_replay.json', reproduction_receipt_sha256=digest(root / 'test_replay.json'))
    monkeypatch.setattr(desktop, 'ROOT', root)
    monkeypatch.setattr(desktop, 'BASE', base)
    monkeypatch.setattr(desktop, 'desktop_directory', lambda: desk)
    result = desktop.deliver_twenty(final, reproduction)
    assert result['status'] == 'COMPLETE'
    assert result['full_ZIP_bytes'] == full.stat().st_size
    assert result['twenty_ZIP_bytes'] == Path(result['Desktop_twenty_file_ZIP']).stat().st_size
    assert result['twenty_file_verification']['exact_file_count'] == 20
    assert result['previous_NEXT7_Desktop_twenty_files_bitwise_unchanged']
    assert json.loads((base / 'DELIVERY_RECEIPT_NEXT8.json').read_text()) == result
