"""Exactly twenty important files and two verified Desktop archives."""
from pathlib import Path
import ctypes
import hashlib
import shutil
import zipfile
from src.next8.common import ROOT, BASE, read_json, digest, atomic, utc_now
from src.next8.research_reports import table, csv_rows
from src.next8.archive_reproduction import stamp_kst


def desktop_directory():
    target = ctypes.create_unicode_buffer(32768)
    result = ctypes.windll.shell32.SHGetFolderPathW(None, 0x10, None, 0, target)
    assert result == 0 and target.value, 'Windows Desktop known folder resolution failed'
    path = Path(target.value).resolve()
    assert path.is_dir()
    return path


def copy_new(source, target):
    source, target = Path(source), Path(target)
    assert not target.exists()
    with source.open('rb') as original, target.open('xb') as output:
        shutil.copyfileobj(original, output, length=1 << 20)
    assert digest(source) == digest(target)


def verify_twenty(directory, archive):
    directory, archive = Path(directory), Path(archive)
    paths = sorted(directory.iterdir())
    assert len(paths) == 20 and all(p.is_file() and not p.is_symlink() for p in paths)
    assert [p.name[:2] for p in paths] == [f'{i:02d}' for i in range(20)]
    hashes = {p.name: digest(p) for p in paths}
    with zipfile.ZipFile(archive) as stream:
        names = stream.namelist()
        assert len(names) == 20 and len(set(names)) == 20 and set(names) == set(hashes)
        assert stream.testzip() is None
        for name, sha in hashes.items():
            assert hashlib.sha256(stream.read(name)).hexdigest() == sha, name
    return {'status': 'PASS', 'exact_file_count': 20, 'files_sha256': hashes,
            'matching_ZIP_exactly_same_twenty_files': True, 'matching_ZIP_sha256': digest(archive)}


def deliver_twenty(final_directory, reproduction):
    final_directory = Path(final_directory)
    assert reproduction['status'] == 'PASS' and reproduction['kind'] == 'FINAL'
    assert reproduction['all_frozen_synthetic_vector_bits_equal']
    full_zip = ROOT / reproduction['archive_path']
    assert digest(full_zip) == reproduction['archive_sha256']
    prior = read_json(BASE / 'RESTORED_NEXT7_AUTHORITY.json')
    for name, sha in prior['previous_Desktop_files_sha256'].items():
        assert digest(Path(prior['previous_Desktop_directory']) / name) == sha
    desktop = desktop_directory()
    folder = desktop / ('StructuralBreak_NEXT8_핵심결과20_' + stamp_kst())
    assert folder.resolve().parent == desktop and not folder.exists()
    archive = desktop / (folder.name + '.zip')
    full_destination = desktop / full_zip.name
    assert not archive.exists() and not full_destination.exists()
    folder.mkdir(exist_ok=False)
    mapping = {
        '00_먼저읽기_NEXT8_KO.md': '00_READ_FIRST_NEXT8_KO.md',
        '01_GPT_HANDOFF_NEXT8.md': '01_GPT_HANDOFF_NEXT8.md',
        '03_FEATURE_CLASSIFICATION_NEXT8.csv': 'FEATURE_CLASSIFICATION_NEXT8.csv',
        '04_VARIABLE_CERTIFICATION_NEXT8.json': 'VARIABLE_CERTIFICATION_REGISTRY.json',
        '05_TARGET_DIRECTION_ATLAS_NEXT8.csv': 'TARGET_DIRECTION_ATLAS.csv',
        '06_CONDITIONAL_DIRECTION_RESULTS_NEXT8.csv': 'CONDITIONAL_DIRECTION_RESULTS.csv',
        '07_MOVEMENT_FORECAST_RESULTS_NEXT8.csv': 'MOVEMENT_FORECAST_RESULTS.csv',
        '09_PREDICTIVE_FACTOR_GRAPH_NEXT8.json': 'PREDICTIVE_FACTOR_GRAPH.json',
        '10_MODEL_COMPARISON_NEXT8.csv': 'MODEL_COMPARISON_NEXT8.csv',
        '11_PAIRED_UNCERTAINTY_NEXT8.csv': 'PAIRED_UNCERTAINTY_NEXT8.csv',
        '14_FEATURE_EXPLANATION_CARDS_KO.md': 'FEATURE_EXPLANATION_CARDS_KO.md',
        '17_EXPERIMENT_LEDGER_NEXT8.csv': 'EXPERIMENT_LEDGER_NEXT8.csv',
        '19_NEXT_SESSION_PLAN_NEXT8.md': 'NEXT_SESSION_PLAN_NEXT8.md'}
    for target, source in mapping.items():
        copy_new(final_directory / source, folder / target)
    policy_names = ('NEXT8_PROTOCOL_LOCK.json', 'NEXT8_FEATURE_BUDGET.json', 'NEXT8_MODEL_MENU_LOCK.json',
                    'FINAL_MODEL_IMPLEMENTATION_POLICY_NEXT8.json', 'CERTIFICATION_AND_SCORING_POLICY_NEXT8.json',
                    'SELF_HISTORY_CENTER_POLICY_NEXT8.json', 'NEXT8_PROTOCOL_SUPPLEMENT_H_CONTEXT.json')
    atomic(folder / '02_PROTOCOL_BUDGET_AND_PROCESS_NEXT8.json', {'status': 'COMPLETE',
        'policies': {name: read_json(BASE / name) for name in policy_names},
        'bottleneck_reviews': [read_json(p) for p in sorted((BASE / 'bottleneck_reviews').glob('REVIEW_*.json'))],
        'fit_accounting': read_json(final_directory / 'FIT_ACCOUNTING_NEXT8.json'),
        'time_accounting': read_json(final_directory / 'TIME_ACCOUNTING_NEXT8.json'),
        'source_manifest': read_json(final_directory / 'SOURCE_MANIFEST_NEXT8.json')})
    for target, inputs in (
        ('08_MOVEMENT_ABLATIONS_AND_CENTER_NEXT8.csv', ('MOVEMENT_FACTOR_ABLATIONS.csv', 'SELF_HISTORY_CENTER_CONTROL_NEXT8.csv')),
        ('12_TIME_AND_H_GENERALIZATION_NEXT8.csv', ('TIME_GENERALIZATION_NEXT8.csv', 'H_BACKGROUND_RESULTS_NEXT8.csv')),
        ('13_COUNTEREXAMPLES_AND_NEGATIVE_CONTROLS_NEXT8.csv', ('COUNTEREXAMPLE_RESULTS_NEXT8.csv', 'NEGATIVE_CONTROLS_NEXT8.csv'))):
        rows = [{'source_table': name, **r} for name in inputs for r in csv_rows(final_directory / name)]
        table(folder / target, rows, ('source_table',))
    copy_new(BASE / 'FIGURES/NEXT8_EIGHT_FIGURES.pdf', folder / '15_NEXT8_EIGHT_FIGURES.pdf')
    atomic(folder / '16_FINAL_AND_LEAKAGE_AUDIT_NEXT8.json', {'status': 'PASS',
        'final_audit': read_json(final_directory / 'FINAL_AUDIT_NEXT8.json'),
        'leakage_audit': read_json(final_directory / 'LEAKAGE_AUDIT_NEXT8.json'),
        'data_access': read_json(final_directory / 'NEXT8_DATA_ACCESS_AUDIT.json'),
        'final_publication_ZIP_actual_replay': reproduction})
    actual_receipt = ROOT / reproduction['reproduction_receipt_path']
    assert digest(actual_receipt) == reproduction['reproduction_receipt_sha256']
    atomic(folder / '18_LATENCY_DETERMINISM_AND_ZIP_REPRODUCTION_NEXT8.json', {'status': 'PASS',
        'latency': read_json(final_directory / 'LATENCY_NEXT8.json'),
        'determinism': read_json(final_directory / 'DETERMINISM_NEXT8.json'),
        'core_reproduction': read_json(final_directory / 'REPRODUCTION_NEXT8.json'),
        'final_ZIP_reproduction': reproduction, 'actual_fresh_process_receipt': read_json(actual_receipt),
        'full_ZIP_Desktop_filename': full_destination.name,
        'self_history_interpretation_limit_KO': (final_directory / 'SELF_HISTORY_BASELINE_LIMIT_KO.md').read_text(encoding='utf-8-sig')})
    paths = sorted(folder.iterdir())
    assert len(paths) == 20
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as stream:
        for path in paths:
            stream.write(path, path.name)
    twenty = verify_twenty(folder, archive)
    copy_new(full_zip, full_destination)
    assert digest(full_destination) == reproduction['archive_sha256']
    for name, sha in prior['previous_Desktop_files_sha256'].items():
        assert digest(Path(prior['previous_Desktop_directory']) / name) == sha
    result = {'status': 'COMPLETE', 'timestamp': utc_now(), 'Desktop_directory': str(folder),
        'Desktop_twenty_file_ZIP': str(archive), 'Desktop_full_NEXT8_ZIP': str(full_destination),
        'twenty_file_verification': twenty, 'full_ZIP_sha256': digest(full_destination),
        'full_ZIP_bytes': full_destination.stat().st_size, 'twenty_ZIP_bytes': archive.stat().st_size,
        'full_ZIP_actual_fresh_replay': reproduction,
        'previous_NEXT7_Desktop_twenty_files_bitwise_unchanged': True,
        'new_directory_and_archives_created_without_overwriting_existing_files': True,
        'all_three_requested_deliveries_complete': True,
        'receipt_self_and_archive_self_hash_cycles_avoided': True}
    atomic(BASE / 'DELIVERY_RECEIPT_NEXT8.json', result)
    return result
