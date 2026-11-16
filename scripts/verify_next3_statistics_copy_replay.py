"""Actually execute the compact-statistics CLI from a minimal ordinary copy."""
from pathlib import Path
import datetime
import os
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next3.artifacts import BASE, atomic_json, digest, progress, read_json, utc_now


def main():
    protocol_path=BASE/'maintenance/CONDITIONAL_STATISTICS_REPLAY_PROTOCOL.json'
    implementation_path=BASE/'maintenance/CONDITIONAL_STATISTICS_REPLAY_IMPLEMENTATION_LOCK.json'
    protocol=read_json(protocol_path); implementation=read_json(implementation_path)
    assert implementation['protocol_sha256'] == digest(protocol_path)
    assert protocol['D002_result_sha256'] == digest(BASE/'selection_uncertainty/RESULTS.json')
    assert protocol['aggregate_contrast_file_sha256'] == digest(BASE/'selection_uncertainty/AGGREGATE_CONTRAST_DRAWS.npz')
    assert protocol['D007_result_sha256'] == digest(BASE/'pair_matrix_audit/run_v1/RESULTS.json')
    for name,sha in implementation['source_sha256'].items(): assert digest(ROOT/name) == sha
    out=BASE/'maintenance/statistics_copy_replay_v1'; out.mkdir(parents=True,exist_ok=False)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    workspace=ROOT.parent/('SB_NEXT3_MAINT_STAT_REPLAY_'+stamp); workspace.mkdir(exist_ok=False)
    payload=workspace/'payload'; payload.mkdir()
    files=['scripts/replay_next3_conditional_statistics.py','src/__init__.py','src/next3/__init__.py',
        'src/next3/artifacts.py','src/next3/archive_contract.py','src/next3/portable_report_contract.py',
        'src/next3/claim_checks.py','src/next3/conditional_statistics.py','requirements-lock.txt']
    files += ['artifacts/next3/selection_uncertainty/'+name for name in ('RESULTS.json','POLICY.json','CONDITIONAL_POOL_ENVELOPE.csv','AGGREGATE_CONTRAST_DRAWS.npz')]
    files += ['artifacts/next3/pair_matrix_audit/run_v1/'+name for name in ('RESULTS.json','CONDITIONAL_SUMMARY_RECONSTRUCTED.csv')]
    manifest={}
    for name in files:
        source=ROOT/name; assert source.is_file() and not source.is_symlink() and not source.is_junction()
        destination=payload/name; destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,destination); assert digest(source) == digest(destination)
        manifest[name]={'sha256':digest(source),'bytes':source.stat().st_size}
    assert len(manifest) == 15 and not (payload/'data').exists()
    assert not any(path.suffix.lower() in ('.npy','.parquet','.sbnext2','.joblib') for path in payload.rglob('*') if path.is_file())
    atomic_json(payload/'SOURCE_MANIFEST_NEXT3.json',{'status':'MAINTENANCE_STATISTICS_COPY_NOT_FINAL_ARCHIVE',
        'created_utc':utc_now(),'files':manifest,'scope':'Only 15 source/lock/aggregate-evidence files; no models, raw inputs or saved point predictions.'},immutable=True)
    command=[sys.executable,'-I','-u',str(payload/'scripts/replay_next3_conditional_statistics.py'),'--output','STATISTICS_REVIEW_RESULT.json']
    atomic_json(out/'START.json',{'created_utc':utc_now(),'workspace':str(workspace),'command':command,
        'copied_files':len(manifest),'fresh_environment_claimed':False,'script_sha256':digest(Path(__file__))},immutable=True)
    with (out/'RUN.log').open('x',encoding='utf-8') as stream:
        result=subprocess.run(command,cwd=payload,env=dict(os.environ,CRUNCH_AUTO_MONKEY_PATCH='false'),
            stdout=stream,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if result.returncode: raise RuntimeError('Statistics-copy replay failed; preserve workspace and logs')
    report=read_json(payload/'STATISTICS_REVIEW_RESULT.json')
    assert report['status'] == 'PASS_EXISTING_CONDITIONAL_STATISTICS_REPLAY' and report['numeric_cross_checks'] == 169
    assert report['candidate_count'] == 12 and report['existing_draw_rows'] == 2000
    assert report['new_fits'] == report['new_Monte_Carlo_draws'] == report['new_point_prediction_or_quality_evaluations'] == 0
    for name,item in manifest.items(): assert digest(ROOT/name) == digest(payload/name) == item['sha256']
    for source,name in [(payload/'STATISTICS_REVIEW_RESULT.json','CHILD_RESULT.json'),(payload/'SOURCE_MANIFEST_NEXT3.json','INPUT_MANIFEST.json')]:
        shutil.copyfile(source,out/name); assert digest(source) == digest(out/name)
    atomic_json(out/'RESULTS.json',{'status':'PASS_EXISTING_STATISTICS_FROM_MINIMAL_COPY','created_utc':utc_now(),
        'workspace':str(workspace),'copied_files':15,'project_modules_loaded_from_copy':len(report['project_modules']),
        'numeric_cross_checks':169,'candidate_count':12,'existing_draw_rows':2000,
        'new_fits':0,'new_Monte_Carlo_draws':0,'new_point_prediction_or_quality_evaluations':0,
        'raw_model_and_saved_point_array_files_absent':True,'all_copied_source_and_inputs_unchanged':True,
        'fresh_environment_claimed':False,'final_archive_bytes_verified':False,'original_project_read_denial_claimed':False,
        'source_sha256':{name:digest(ROOT/name) for name in ('scripts/verify_next3_statistics_copy_replay.py','scripts/replay_next3_conditional_statistics.py','src/next3/conditional_statistics.py')},
        'files_sha256':{path.name:digest(path) for path in out.iterdir() if path.is_file()},
        'protocol_sha256':digest(protocol_path),'implementation_lock_sha256':digest(implementation_path),
        'scope':'The actual reviewer CLI ran from 15 ordinary copied files using the existing pinned environment. No models/raw/point predictions, no new bootstrap draws, fits or quality evaluations. Existing conditional evidence only.'},immutable=True)
    progress('Conditional evidence portable replay passed','169 existing numeric summaries verified from 15 ordinary source/aggregate files; no models/raw/point arrays/new draws/fits')
    print('NEXT3 STATISTICS COPY REPLAY PASS 169 NUMERIC CHECKS',flush=True)


if __name__ == '__main__': main()
