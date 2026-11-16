"""Create, extract, reproduce and copy the final NEXT7 handoff into20 desktop files."""
from pathlib import Path
import argparse
import datetime as dt
import os
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.next7.common import BASE,read_json,save,digest,utc_now,activity,note
from src.next7.handoff_archive import checked_name,checked_source,create_archive,extract_verified,SAFE_NUMPY_FILES
from src.next7.worker_environment import environment


def add_direct(files,directory,suffixes):
    assert directory.is_dir(),str(directory)
    files.update(p.relative_to(ROOT).as_posix() for p in directory.iterdir() if p.is_file() and p.suffix in suffixes)


def verify_hashes(directory,mapping):
    checked=0
    for name,sha in mapping.items():
        path=(directory/name).resolve(strict=True)
        assert path.is_relative_to(directory.resolve()),'Evidence reference escaped its recorded root'
        assert digest(path)==sha,'Evidence bytes changed after completion: '+str(path)
        checked+=1
    return checked


def verify_receipt_contents(directory,receipt_name='DONE.json'):
    receipt=read_json(directory/receipt_name)
    assert receipt['status'] in ('PASS','COMPLETE')
    checked=0
    for key in ('files_sha256','source_files_sha256','outputs_sha256'):
        if isinstance(receipt.get(key),dict):checked+=verify_hashes(directory,receipt[key])
    for key in ('input_sha256','inputs_sha256'):
        if isinstance(receipt.get(key),dict):checked+=verify_hashes(BASE,receipt[key])
    if receipt.get('summary_sha256'):
        checked+=verify_hashes(directory,{'PER_H_SUPPORT.csv':receipt['summary_sha256']})
    assert checked,'A completed payload receipt must bind concrete input/output bytes'
    return checked


def verify_closed_research_evidence():
    ready=read_json(BASE/'RESEARCH_COMPLETION_READY.json')
    assert ready['status']=='PASS' and ready['outstanding_scientific_tasks']==0
    checked=verify_hashes(BASE,ready['required_artifacts_sha256'])
    for kind in ('GRADIENTS','CLASSIFIERS','CONDITIONAL'):
        receipt=read_json(BASE/f'COMPLETE_DIAGNOSTICS_{kind}_NEXT7.json')
        assert receipt['status']=='COMPLETE' and not receipt['partial_cohorts_included']
        checked+=verify_hashes(BASE,receipt['inputs_sha256'])
        checked+=verify_hashes(BASE,receipt['outputs_sha256'])
    return checked


def verify_prior_desktop():
    path=BASE/'audit/NEXT6_DESKTOP_PRESERVATION_BASELINE.json'
    baseline=read_json(path)
    directory=Path(baseline['directory'])
    assert directory.parent.resolve()==(Path.home()/'Desktop').resolve()
    assert directory.name.startswith('StructuralBreak_NEXT6_') and baseline['file_count']==20
    assert {p.name for p in directory.iterdir()}==set(baseline['files'])
    for name,expected in baseline['files'].items():
        p=directory/name
        assert p.is_file() and p.stat().st_size==expected['bytes'] and digest(p)==expected['sha256']
    return digest(path)


def inventory(diagnostics,normalizers):
    plan=read_json(BASE/'HANDOFF_CONTENT_PLAN_NEXT7.json')
    files={'artifacts/next7/'+name for name in plan['explicit_root_artifact_names']}
    for amendment_path in sorted(BASE.glob('HANDOFF_CONTENT_PLAN_AMENDMENT_*.json')):
        amendment=read_json(amendment_path)
        assert amendment['base_plan_sha256']==digest(BASE/'HANDOFF_CONTENT_PLAN_NEXT7.json')
        files.add(amendment_path.relative_to(ROOT).as_posix())
        files.update('artifacts/next7/'+name for name in amendment['additional_root_artifact_names'])
    files.update(plan['root_source_files'])
    files.update('configs/'+name for name in plan['required_config_files'])
    files.update(p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py') if '__pycache__' not in p.parts)
    files.update(p.relative_to(ROOT).as_posix() for p in (ROOT/'scripts/next7').glob('*.py'))
    files.update(p.relative_to(ROOT).as_posix() for p in (ROOT/'scripts/next7').glob('*.mjs'))
    files.update(p.relative_to(ROOT).as_posix() for p in (ROOT/'src/next7').glob('*.html'))
    for fold in range(5):
        directory=BASE/f'research_bundles/fold_{fold}'
        assert len(list(directory.iterdir()))==7
        add_direct(files,directory,{'.json','.joblib'})
    add_direct(files,BASE/'reproduction_fixture',{'.json','.npz'})
    add_direct(files,BASE/'frozen_reference_fixture',{'.json','.npz'})
    files.update(read_json(BASE/'FROZEN_REFERENCE_DELIVERY_LOCK_NEXT7.json')['files_sha256'])
    add_direct(files,BASE/'research_notes',{'.md'})
    add_direct(files,BASE/'bottlenecks',{'.json'})
    add_direct(files,BASE/'synthetic_stats',{'.json','.csv'})
    files.update(SAFE_NUMPY_FILES)
    for generation in ('g1','g2','g3'):
        for directory in (BASE/'bootstrap'/generation).iterdir():
            if directory.is_dir():add_direct(files,directory,{'.json','.npz'})
    for group in ('classifier_statistics','conditional_statistics','support_partition',
                  'crossed_H_length_comparison','normalizer_transport/correlations','normalizer_transport/H_parameters'):
        for done_path in (BASE/group).rglob('DONE.json'):
            verify_receipt_contents(done_path.parent)
            add_direct(files,done_path.parent,{'.json','.csv','.npz'})
    for done_path in (BASE/'bootstrap_support').glob('*/*/DONE.json'):
        verify_receipt_contents(done_path.parent)
        add_direct(files,done_path.parent,{'.json','.csv'})
    for fold in range(5):
        add_direct(files,BASE/f'g3_full_gradient/fold_{fold}',{'.json','.csv'})
    for directory in (BASE/'visuals/v3', BASE/'visuals/v4', BASE/'visuals/diagnostics'/diagnostics,
                      BASE/'visuals/normalizer_transport'/normalizers,
                      BASE/'visuals/classifier_nulls/four_backgrounds_001',
                      BASE/'visuals/scale_evidence/original_cases_001',
                      BASE/'visuals/classifier_scale/original_cases_002',
                      BASE/'visuals/classifier_stress/full_suite_001'):
        receipt_name=next(name for name in ('BUILD_DONE.json','MANIFEST.json','DONE.json') if (directory/name).is_file())
        verify_receipt_contents(directory,receipt_name)
        add_direct(files,directory,{'.json','.csv','.npz','.png','.svg','.html'})
    # Audit summaries and test XML contain no raw competition arrays. Log files,
    # test fixture arrays, browser profiles and process checkpoints are omitted.
    files.update(p.relative_to(ROOT).as_posix() for p in (BASE/'audit').rglob('*')
                 if p.is_file() and p.suffix in ('.json','.xml') and 'owned_crash' not in p.parts)
    role=read_json(BASE/'FINAL_CANDIDATE_FREEZE_NEXT7.json')['selected_final_primary']
    assert digest(ROOT/role['model_file'])==role['model_sha256']
    files.add(role['model_file'])
    if (ROOT/role['entry']).is_file():files.add(role['entry'])
    import numpy as np
    for name in SAFE_NUMPY_FILES:
        array=np.load(ROOT/name,allow_pickle=False)
        expected=(2000,512) if name.endswith('COUNTS_2000x512.npy') else (2001,)
        assert array.shape==expected and array.dtype.kind in 'uif' and np.isfinite(array).all()
        if len(expected)==1:assert np.all((array>=0)&(array<=1))
    for name in files:checked_source(ROOT,checked_name(name))
    return sorted(files)


def main(snapshot,diagnostics,normalizers):
    for label in (snapshot,diagnostics,normalizers):
        assert label and all(c.isalnum() or c in '_-' for c in label)
    final=read_json(BASE/'FINAL_AUDIT_NEXT7.json')
    assert final['status']=='PASS' and final['decision']=='NEXT7_NO_PROMOTION'
    assert final['productive_research_seconds']>=36000 and final['final_audit_seconds']>0
    assert final['final_audit_target_seconds']==3600 and final['final_audit_target_is_approximate']
    for mapping in ('audit_evidence_sha256','audit_outputs_sha256'):
        verify_hashes(BASE,final[mapping])
    closed_evidence_files=verify_closed_research_evidence()
    documents=read_json(BASE/'audit/FINAL_DOCUMENTS_DONE.json')
    assert documents['status']=='COMPLETE' and documents['answered_user_questions']==45
    assert documents['final_audit_sha256']==digest(BASE/'FINAL_AUDIT_NEXT7.json')
    verify_hashes(BASE,documents['files_sha256'])
    for name,sha in read_json(BASE/'SOURCE_MANIFEST_NEXT7.json')['files_sha256'].items():
        assert digest(ROOT/name)==sha,'Source/model bytes changed after final audit: '+name
    assert digest(BASE/'FINAL_CANDIDATE_FREEZE_NEXT7.json')==final['candidate_freeze_sha256']
    prior_desktop_baseline_sha=verify_prior_desktop()
    stop=read_json(BASE/'runtime/STOP_SCHEDULER.json')
    assert stop['scope']=='FINAL_DELIVERY_LEDGER_FREEZE', 'A maintenance restart is not final ledger closure'
    from src.next7.queue import identity_alive
    scheduler_owner=read_json(BASE/'runtime/SCHEDULER_OWNER.json')
    assert not identity_alive(scheduler_owner['pid'],scheduler_owner['process_created']), 'Wait for the final monitor tick and clean exit before exporting mutable ledgers'
    heartbeat=read_json(BASE/'runtime/HEARTBEAT.json')
    assert not heartbeat['queue_runner_pids'] and not heartbeat['hang_suspected']
    # Every final file is already concrete; packaging never creates a missing audit.
    plan=read_json(BASE/'HANDOFF_CONTENT_PLAN_NEXT7.json')
    assert all((BASE/name).is_file() for name in plan['required_before_export'])
    for candidate in ('FINAL_CANDIDATE_FREEZE_NEXT7.json','RESEARCH_COMPLETION_READY.json'):
        assert (BASE/candidate).is_file()
    relative='delivery/'+snapshot
    directory=BASE/relative
    directory.mkdir(parents=True,exist_ok=False)
    archive=BASE/f'STRUCTURAL_BREAK_GPT_HANDOFF_NEXT7_{snapshot}.zip'
    files=inventory(diagnostics,normalizers)
    save(relative+'/EXPORT_ALLOWLIST.json',{'status':'FROZEN','created_utc':utc_now(),
        'source_sha256':digest(__file__),'archive_library_sha256':digest(ROOT/'src/next7/handoff_archive.py'),
        'content_plan_sha256':digest(BASE/'HANDOFF_CONTENT_PLAN_NEXT7.json'),'files':files,
        'file_count':len(files),'uncompressed_bytes':sum((ROOT/name).stat().st_size for name in files),
        'closed_research_evidence_hash_checks':closed_evidence_files,
        'completed_statistics_and_visual_manifest_bytes_rechecked':True,
        'archived_roles':'Retained BL80 primary and preexisting D2/J102 references; three negative research recipes only as fivefold reproduction evidence.'})
    with activity('Final handoff ZIP byte verification and extracted replay','FINAL_AUDIT','FINAL_AUDIT'):
        manifest=create_archive(ROOT,files,archive,{'created_utc':utc_now(),'decision':'NEXT7_NO_PROMOTION',
            'final_computation_audit_sha256':digest(BASE/'FINAL_AUDIT_NEXT7.json'),
            'source_manifest_sha256':digest(BASE/'SOURCE_MANIFEST_NEXT7.json'),
            'export_allowlist_sha256':digest(directory/'EXPORT_ALLOWLIST.json'),
            'external_delivery_receipt':'The archive SHA and fresh extracted replay result are recorded outside this archive to avoid a self-hash cycle.'})
        archive_sha=digest(archive)
        extracted=directory/'extracted'
        recovered=extract_verified(archive,extracted)
        assert recovered==manifest
        script='scripts/next7/reproduce_research_bundle.py'
        fixture=read_json(extracted/'artifacts/next7/reproduction_fixture/FIXTURE.json')
        assert digest(extracted/script)==fixture['source_sha256']
        env=environment()
        env.pop('PYTHONPATH',None)
        env['PYTHONDONTWRITEBYTECODE']='1'
        env['NUMBA_CACHE_DIR']=str(extracted/'artifacts/next7/numba_cache')
        command=[sys.executable,'-B',script,'--output-directory',str(extracted/'demo_output'),
                 '--deny-original-root',str(ROOT)]
        with (directory/'EXTRACTED_REPRODUCTION.log').open('x',encoding='utf-8') as output:
            process=subprocess.run(command,cwd=extracted,env=env,stdout=output,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        assert process.returncode==0,'Extracted reproduction failed; preserve archive and inspect its log'
        replay=read_json(extracted/'demo_output/REPRODUCTION_DONE.json')
        assert replay['status']=='PASS' and replay['original_workspace_access_blocked']
        assert replay['denied_access_attempts']==0 and len(replay['checks'])==90
        assert all(row['bitwise_equal'] and row['max_absolute_error']==0 for row in replay['checks'])
        reference_command=[sys.executable,'-B','scripts/next7/reproduce_frozen_references.py',
            '--output-directory',str(extracted/'demo_reference_output'),'--deny-original-root',str(ROOT)]
        with (directory/'EXTRACTED_REFERENCE_REPRODUCTION.log').open('x',encoding='utf-8') as output:
            reference_process=subprocess.run(reference_command,cwd=extracted,env=env,stdout=output,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        assert reference_process.returncode==0,'Historical reference extracted replay failed; preserve its log'
        reference_replay=read_json(extracted/'demo_reference_output/FROZEN_REFERENCE_REPRODUCTION_DONE.json')
        assert reference_replay['status']=='PASS' and reference_replay['original_workspace_access_blocked']
        assert reference_replay['public_entry_paths']==9 and reference_replay['scalar_predictions']==9216
        assert reference_replay['denied_access_attempts']==0
        assert all(row['bitwise_equal'] and row['max_absolute_error']==0 for row in reference_replay['checks'])
        assert digest(archive)==archive_sha
        save(relative+'/ARCHIVE_REPRODUCTION_DONE.json',{'status':'PASS','created_utc':utc_now(),
            'archive':archive.name,'archive_sha256':archive_sha,'archive_bytes':archive.stat().st_size,
            'every_archived_file_byte_verified':True,'archived_file_count':len(files),
            'isolated_fresh_child':True,'raw_competition_data_included':False,
            'original_project_read_attempts':0,'paths_checked':90,'scalar_outputs_checked':92160,
            'all_scalar_outputs_bitwise_equal':True,'max_absolute_error':0.,
            'extracted_reproduction_receipt_sha256':digest(extracted/'demo_output/REPRODUCTION_DONE.json'),
            'historical_reference_public_entry_paths':9,'historical_reference_scalar_outputs':9216,
            'historical_reference_all_outputs_bitwise_equal':True,
            'historical_reference_replay_receipt_sha256':digest(extracted/'demo_reference_output/FROZEN_REFERENCE_REPRODUCTION_DONE.json'),
            'computation_audit_sha256':digest(BASE/'FINAL_AUDIT_NEXT7.json')})
        desktop=Path.home()/'Desktop'
        assert desktop.is_dir()
        target=desktop/f'StructuralBreak_NEXT7_핵심결과20_{snapshot}'
        target.mkdir(exist_ok=False)
        network_image=BASE/'visuals/v4/NETWORK_CLASSIFIER_recovery_with_memory.png'
        assert network_image.is_file(),'Use the completed recovery illustration'
        mapping=[('01_먼저읽기_NEXT7_KO.md',BASE/'00_READ_FIRST_NEXT7_KO.md'),
            ('02_GPT_HANDOFF_NEXT7.md',BASE/'01_GPT_HANDOFF_NEXT7.md'),
            ('03_MODEL_COMPARISON_NEXT7.csv',BASE/'MODEL_COMPARISON_NEXT7.csv'),
            ('04_CV_RESULTS_NEXT7.csv',BASE/'CV_RESULTS_NEXT7.csv'),
            ('05_PAIRED_BOOTSTRAP_NEXT7.csv',BASE/'PAIRED_BOOTSTRAP_NEXT7.csv'),
            ('06_CLASSIFIER_LONG_NULL_RESULTS_NEXT7.csv',BASE/'CLASSIFIER_LONG_NULL_RESULTS_NEXT7.csv'),
            ('07_FALSE_LATCH_RESULTS_NEXT7.csv',BASE/'FALSE_LATCH_RESULTS_NEXT7.csv'),
            ('08_SCALE_EXPANSION_CONTRACTION_RESULTS.csv',BASE/'SCALE_EXPANSION_CONTRACTION_RESULTS.csv'),
            ('09_SHOCK_RESULTS_NEXT7.csv',BASE/'SHOCK_RESULTS_NEXT7.csv'),
            ('10_H_CALIBRATION_RESULTS.csv',BASE/'H_CALIBRATION_RESULTS.csv'),
            ('11_EVENT_MEMORY_RESULTS.csv',BASE/'EVENT_MEMORY_RESULTS.csv'),
            ('12_TARGET_ALIGNMENT_RESULTS.csv',BASE/'TARGET_ALIGNMENT_RESULTS.csv'),
            ('13_TIME_FRACTURE_NEXT7.csv',BASE/'TIME_FRACTURE_NEXT7.csv'),
            ('14_BREAK_AGE_NEXT7.csv',BASE/'BREAK_AGE_NEXT7.csv'),
            ('15_BOTTLENECK_HISTORY_NEXT7.csv',BASE/'BOTTLENECK_HISTORY_NEXT7.csv'),
            ('16_LATENCY_NEXT7.csv',BASE/'LATENCY_NEXT7.csv'),
            ('17_FINAL_AUDIT_NEXT7.json',BASE/'FINAL_AUDIT_NEXT7.json'),
            ('18_DYNAMIC_STATE_TRANSITION_NEXT7.html',BASE/'visuals/v4/DYNAMIC_STATE_TRANSITION_NEXT7.html'),
            ('19_RECOVERY_STATE_NETWORK_NEXT7.png',network_image),
            ('20_'+archive.name,archive)]
        copied={}
        for name,source in mapping:
            destination=target/name
            with source.open('rb') as src,destination.open('xb') as dst:shutil.copyfileobj(src,dst)
            assert digest(source)==digest(destination)
            copied[name]={'source':source.relative_to(ROOT).as_posix(),'sha256':digest(destination),'bytes':destination.stat().st_size}
        assert len(list(target.iterdir()))==20 and all(p.is_file() for p in target.iterdir())
        assert verify_prior_desktop()==prior_desktop_baseline_sha
        save('FINAL_DELIVERY_AUDIT_NEXT7.json',{'status':'PASS','created_utc':utc_now(),
            'decision':'NEXT7_NO_PROMOTION','archive':archive.name,'archive_sha256':archive_sha,
            'archive_reproduction_receipt':relative+'/ARCHIVE_REPRODUCTION_DONE.json',
            'archive_reproduction_sha256':digest(directory/'ARCHIVE_REPRODUCTION_DONE.json'),
            'desktop_directory':str(target),'desktop_files':copied,'desktop_file_count':20,
            'copied_only':True,'prior_desktop_folders_preserved':True,
            'previous_NEXT6_desktop20_rechecked_before_and_after_copy':True,
            'previous_desktop_baseline_sha256':prior_desktop_baseline_sha,
            'computation_audit_sha256':digest(BASE/'FINAL_AUDIT_NEXT7.json'),
            'wall_seconds_at_delivery':dt.datetime.now(dt.timezone.utc).timestamp()-read_json(BASE/'SESSION_NEXT7.json')['started_epoch'],
            'delivery_postlude_not_backfilled_into_frozen_productive_or_audit_time':True})
    print('DELIVERED',str(target),'ZIP_SHA256',archive_sha,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--snapshot',required=True)
    parser.add_argument('--diagnostic-snapshot',default='final_all_001')
    parser.add_argument('--normalizer-snapshot',default='final_all_001')
    args=parser.parse_args()
    main(args.snapshot,args.diagnostic_snapshot,args.normalizer_snapshot)
