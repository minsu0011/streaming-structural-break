"""Preserve failed held-out harnesses; keep measured runtime/cold sources unchanged."""
from pathlib import Path
import ast
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, read_json, atomic_json, digest, utc_now, progress
from src.next3.maintenance import register_job


def main():
    base = BASE/'engineering/lean_shared'
    core_path = base/'IMPLEMENTATION_LOCK_CORRECTION_001.json'; core = read_json(core_path)
    for relative, sha in core['source_sha256'].items(): assert digest(ROOT/relative) == sha, relative
    source = (ROOT/'scripts/verify_next3_lean_shared.py').read_text(encoding='utf-8')
    begin = source.index('def PreparedPackedSharedPredictor('); end = source.index('from src.next2.shared_blend_runtime', begin)
    source = source[:begin]+'''def PreparedPackedSharedPredictor(model, *, variant):
    from src.next3.lean_heldout_validation import prepare_heldout_record
    return prepare_heldout_record(dumps(model), variant=variant)
'''+source[end:]
    source = source.replace("OUT = BASE/'engineering/lean_shared'", "OUT = BASE/'engineering/lean_shared/heldout_metadata_corrected_v1'")
    begin = source.index('def checked_implementation():'); end = source.index('\n\ndef checked_model', begin)
    source = source[:begin]+'''def checked_implementation():
    from scripts.verify_next3_lean_shared import checked_implementation as core_check
    core_sha = core_check()
    lock = read_json(BASE/'engineering/lean_shared/VALIDATION_HARNESS_CORRECTION_002.json')
    assert lock['core_implementation_lock_sha256'] == core_sha
    for name, sha in lock['source_sha256'].items(): assert digest(ROOT/name) == sha, name
    return core_sha
'''+source[end:]
    source = source.replace("'stage': stage, 'runtime_sha256'", "'validation_harness_revision': 'unused_C205_ARCH_hash_002', 'stage': stage, 'runtime_sha256'")
    source = source.replace("'stage': 'alternate', 'runtime_sha256'", "'validation_harness_revision': 'unused_C205_ARCH_hash_002', 'stage': 'alternate', 'runtime_sha256'")
    verification = ROOT/'scripts/verify_next3_lean_shared_legacy_metadata.py'
    assert not verification.exists(); ast.parse(source); verification.write_text(source, encoding='utf-8', newline='\n')
    source = (ROOT/'scripts/benchmark_next3_lean_direct.py').read_text(encoding='utf-8')
    source = source.replace('from scripts.verify_next3_lean_shared import checked_implementation',
        'from scripts.verify_next3_lean_shared_legacy_metadata import checked_implementation')
    before = "path = BASE/'engineering/lean_shared'/stage/'RESULTS.json'; report = read_json(path)"
    after = "folder = BASE/'engineering/lean_shared'; folder = folder/'heldout_metadata_corrected_v1' if stage in ('oof', 'alternate') else folder\n        path = folder/stage/'RESULTS.json'; report = read_json(path)"
    assert before in source; source = source.replace(before, after)
    direct = ROOT/'scripts/benchmark_next3_lean_direct_validated.py'
    assert not direct.exists(); ast.parse(source); direct.write_text(source, encoding='utf-8', newline='\n')
    sources = ['src/next3/lean_heldout_validation.py', verification.relative_to(ROOT).as_posix(), direct.relative_to(ROOT).as_posix(),
        'tests/test_next3_lean_heldout_validation.py']
    atomic_json(base/'VALIDATION_HARNESS_CORRECTION_002.json', {'created_utc': utc_now(),
        'core_implementation_lock_sha256': digest(core_path), 'source_sha256': {p: digest(ROOT/p) for p in sources},
        'finding': 'Frozen original/ALT C205 validation bundles omit the unused old_arch_hash; final all-DEV bundle includes it. C205 has exactly 24 dependence inputs and no old ARCH branch.',
        'permitted_metadata_difference': 'Only complement.old_arch_hash may be empty in the validation helper; exact frozen feature names, all live source contracts and canonical class-loader checks remain mandatory.',
        'runtime_decoder_and_public_entry_sources_changed': False, 'existing_cold_measurements_preserved_without_retry': True,
        'original_failed_harness_outputs_preserved': ['oof', 'alternate'], 'new_fits': 0,
        'reason_not_new_model_variant': 'The runtime receives unchanged numeric trees and selectors; this is a validation-only legacy metadata adapter.',
    }, immutable=True)
    # Superseded jobs are explicitly quarantined before a child starts; they are
    # not fabricated successful runs or scientific fit failures.
    for job_id in ('E307_LEAN_PUBLIC_API', 'E308_LEAN_DIRECT'):
        path = BASE/'jobs'/job_id/'STATE.json'; assert not path.exists()
        job = read_json(BASE/'queue'/(job_id+'.json'))
        atomic_json(path, {**job, 'status': 'QUARANTINED', 'attempt': 0,
            'terminal_kind': 'SUPERSEDED_WITHOUT_EXECUTION', 'quarantined_utc': utc_now(),
            'reason': 'Replaced by jobs using separately preserved corrected held-out validation receipts'}, immutable=True)
    register_job({'id': 'E304Z_LEAN_HELDOUT_METADATA_TESTS', 'kind': 'maintenance',
        'argv': ['-m', 'pytest', '-q', 'tests/test_next3_lean_heldout_validation.py', '--junitxml=artifacts/next3/maintenance/LEAN_HELDOUT_METADATA_TESTS.xml'],
        'description': 'Actual frozen OOF metadata compatibility and rejection boundaries; core runtime unchanged', 'retry_safe': False})
    for number, stage in ((305, 'oof'), (306, 'alternate')):
        register_job({'id': f'E{number}A_LEAN_{stage.upper()}_METADATA', 'kind': 'maintenance',
            'argv': ['scripts/verify_next3_lean_shared_legacy_metadata.py', '--stage', stage],
            'description': 'Complete exact held-out parity using validated absence of unused C205 ARCH metadata only', 'retry_safe': False})
    register_job({'id': 'E307A_LEAN_PUBLIC_API', 'kind': 'maintenance',
        'argv': ['scripts/verify_next3_lean_public_api.py'], 'description': 'Unchanged actual lean public API after corrected held-out parity',
        'requires': ['artifacts/next3/engineering/lean_shared/heldout_metadata_corrected_v1/alternate/RESULTS.json'], 'retry_safe': False})
    register_job({'id': 'E308A_LEAN_DIRECT', 'kind': 'maintenance',
        'argv': ['scripts/benchmark_next3_lean_direct_validated.py'], 'description': 'First three paired direct 10M trials; same runtime/thresholds with corrected prerequisite receipt locations',
        'requires': ['artifacts/next3/lean_public_api_all_dev/RESULTS.json'], 'retry_safe': False})
    progress('E003 held-out metadata harness corrected', 'Core runtime/cold measurements unchanged; failed receipts and superseded pending jobs preserved', experiment='E003_LEAN_SHARED')


if __name__ == '__main__': main()
