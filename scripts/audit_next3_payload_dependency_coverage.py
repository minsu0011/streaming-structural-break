"""Check current source-set and review inputs in an actual filtered ordinary copy."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next3.artifacts import BASE, atomic_json, digest, progress, read_json, utc_now
from src.next3.final_source_contract import snapshot_sources, assert_source_snapshot
from src.next3.path_budget import check_windows_copy_path_budget
from scripts.package_next3_handoff import snapshot_payload


def main():
    protocol_path = BASE/'maintenance/PAYLOAD_DEPENDENCY_COVERAGE_PROTOCOL_CORRECTION_002.json'
    protocol = read_json(protocol_path)
    assert protocol['status'] == 'REGISTERED_PACKAGING_MAINTENANCE'
    for name,sha in protocol['source_sha256'].items(): assert digest(ROOT/name) == sha
    sources = snapshot_sources(ROOT)
    out = BASE/'handoff/m3'
    out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'START.json',{'status':'MAINTENANCE_DRY_RUN_ONLY','created_utc':utc_now(),
        'protocol_sha256':digest(protocol_path),'source_set':sources,
        'actual_final_audit':False,'archive_created':False},immutable=True)
    progress('Current payload dependency coverage','Copy actual filtered source/evidence bytes and verify the complete current execution source set; final files remain pending')
    payload = out/'payload'
    files,excluded,missing = snapshot_payload(payload,require_audit=False)
    path_budget = check_windows_copy_path_budget(payload,files)
    for name,record in files.items():
        path = payload/name
        assert path.is_file() and not path.is_symlink() and not path.is_junction()
        assert digest(path) == record['sha256'] and path.stat().st_size == record['bytes']
    missing_sources = sorted(set(sources['files'])-set(files))
    assert not missing_sources, missing_sources
    for name,record in sources['files'].items(): assert files[name] == record
    checked = []
    for name in protocol['reviewer_command_inputs']:
        assert name in files and digest(payload/name) == digest(ROOT/name)
        checked.append({'path':name,'sha256':files[name]['sha256']})
    configuration = read_json(ROOT/'configs/next3/LEAN_RUNTIME_INPUTS.json')['files_sha256']
    for name,sha in configuration.items(): assert files[name]['sha256'] == sha
    assert_source_snapshot(ROOT,sources)
    assert not any(name.startswith('data/') or name.endswith(('.whl','.zip')) for name in files)
    protected = read_json(BASE/'preflight/FROZEN_REFERENCES.json')['protected_sha256']
    original_models = {name:sha for name,sha in protected.items() if name.endswith('.joblib')}
    assert len(original_models) == 4
    assert {name for name in files if name.endswith('.joblib')} == set(original_models)
    for name,sha in original_models.items(): assert files[name]['sha256'] == sha
    atomic_json(out/'SOURCE_COPY_MANIFEST.json',{'status':'MAINTENANCE_COPY_NOT_FINAL_ARCHIVE',
        'files':files,'excluded_optional_records':excluded},immutable=True)
    report = {'status':'PASS_CURRENT_PAYLOAD_DEPENDENCY_COVERAGE','created_utc':utc_now(),
        'files_copied_and_rehashed':len(files),'uncompressed_bytes':sum(v['bytes'] for v in files.values()),
        'windows_path_budget':path_budget,
        'complete_current_execution_sources_covered':sources['file_count'],
        'source_content_set_sha256':sources['content_set_sha256'],'missing_current_execution_sources':[],
        'source_set_unchanged_through_copy':True,'reviewer_command_inputs':checked,
        'frozen_lean_config_inputs_covered':len(configuration),
        'original_protected_joblib_models_preserved_by_SHA_only':original_models,
        'joblib_deserialization_performed':False,
        'protected_old_files':len(read_json(BASE/'preflight/FROZEN_REFERENCES.json')['protected_sha256']),
        'excluded_optional_records':excluded,'required_final_reports_still_missing':missing,
        'source_copy_manifest_sha256':digest(out/'SOURCE_COPY_MANIFEST.json'),
        'copy_directory':payload.relative_to(ROOT).as_posix(),
        'source_sha256':protocol['source_sha256'],'protocol_sha256':digest(protocol_path),
        'new_fits':0,'new_quality_scores':0,'inference_replayed_in_this_check':False,
        'actual_final_archive_created':False,'fresh_raw_reproduction_claimed':False,
        'default_final_reviewer_command_ready_claimed':False,
        'scope':'Actual current ordinary source/evidence snapshot and complete current code/config coverage. Pending final deployment descriptors and fresh models are not substituted; final source lock, full tests, raw reproduction, final snapshot and actual ZIP replays remain mandatory.'}
    atomic_json(out/'RESULTS.json',report,immutable=True)
    atomic_json(BASE/'maintenance/PAYLOAD_DEPENDENCY_COVERAGE_RESULT.json',report,immutable=True)
    progress('Current payload dependency coverage passed',
        f'{len(files)} ordinary files rehashed; all {sources["file_count"]} current execution inputs covered; no raw files or final-delivery claim')
    print('NEXT3 CURRENT PAYLOAD COVERAGE PASS',len(files),sources['file_count'],flush=True)


if __name__ == '__main__': main()
