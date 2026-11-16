"""Build a fresh raw-free archive and actually replay its ten fitted procedures."""
from pathlib import Path
import sys
import os
import argparse
import subprocess
import shutil
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now

def main(tag,final):
    assert tag.isalnum() and len(tag)<=8
    admission=read_json(BASE/'runtime/FINAL_AUDIT_ADMISSION_NEXT9.json');assert admission['status']=='PASS'
    from src.next9.delivery_payload_v5 import select_payload,materialize,zip_payload,verify_extract
    out=BASE/'delivery'/tag
    if out.exists():raise FileExistsError('Preserve previous attempt; use a new short tag')
    out.mkdir(parents=True)
    with activity('Raw-free archive assembly, CRC, manifest and fresh extraction '+tag,'FINAL_AUDIT','B9_NORMALIZATION_STATE_AVAILABILITY'):
        paths=select_payload(final=final);content=materialize(paths,out/'p')
        archive=out/f'NEXT9_{tag}.zip';archive_sha=zip_payload(out/'p',archive)
        extraction=out/'x';verification=verify_extract(archive,extraction)
        atomic(out/'ARCHIVE_INTEGRITY.json',dict(**verification,archive_bytes=archive.stat().st_size))
    env=dict(os.environ);env.update(PYTHONPATH=str(extraction),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',NEXT9_PORTABLE_REPLAY='1')
    for key in list(env):
        if key.startswith('NEXT5_TEST_'):del env[key]
    commands=[]
    for fold in range(5):
        for mode in ['NORMAL','EARLY']:
            commands.append((f'F{fold}_{mode}',[str(extraction/'scripts/next9/replay_next9_synthetic_portable_v6.py'),
                '--fold',str(fold),'--mode',mode,'--verify','--original',str(ROOT)]))
    commands.append(('PUBLIC_ENTRIES',[str(extraction/'scripts/next9/replay_next9_public_entries_v5.py'),'--original',str(ROOT)]))
    reports=[]
    extracted_base=extraction/BASE.relative_to(ROOT)
    for name,arguments in commands:
        command=[str(ROOT/'.venv/Scripts/python.exe'),'-X','utf8','-B',*arguments]
        with (out/(name+'.log')).open('x',encoding='utf-8') as log:
            process=subprocess.Popen(command,cwd=extraction,env=env,stdout=log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            import psutil
            atomic(out/(name+'_START.json'),dict(timestamp=utc_now(),pid=process.pid,created=psutil.Process(process.pid).create_time(),
                command=command,archive_sha256=archive_sha,process_created_by_this_reproduction=True))
            code=process.wait()
        # Preserve original child CPU-backed intervals with exact bytes. Parent
        # wait is not credited, and overlapping work is never summed.
        copied=[]
        for path in (extracted_base/'ledger').glob('*.json'):
            record=read_json(path)
            if record['kind']!='FINAL_AUDIT' or record['started_epoch']<admission['admitted_epoch']:continue
            job=extracted_base/'runtime/jobs'/path.name
            if not job.is_file():continue
            assert digest(path)==digest(job)
            for source,target in [(path,BASE/'ledger'/path.name),(job,BASE/'runtime/jobs'/path.name)]:
                if target.exists():assert digest(target)==digest(source)
                else:shutil.copyfile(source,target)
            copied.append(dict(id=record['id'],sha256=digest(path),status=record['status']))
        atomic(out/(name+'_DONE.json'),dict(timestamp=utc_now(),returncode=code,imported_child_activity=copied,
            log_sha256=digest(out/(name+'.log')),parent_wait_not_credited=True))
        assert code==0,'Fresh ZIP replay failed; original logs and artifacts preserved: '+name
        reports.append(name)
        print(utc_now(),'Fresh archive completed',tag,name,flush=True)
    with activity('Verify all delivered member bytes after actual fresh-model replay '+tag,'FINAL_AUDIT','B9_NORMALIZATION_STATE_AVAILABILITY'):
        for name,sha in content.items():assert digest(extraction/name)==sha,name
        scopes=[]
        for fold in range(5):
            for mode in ['NORMAL','EARLY']:
                path=extracted_base/'portable_verification'/f'outer_{fold}'/mode/'DONE.json'
                record=read_json(path);assert record['status']=='PASS' and record['cases']==168 and record['points']==86016
                scopes.append(dict(fold=fold,mode=mode,receipt_sha256=digest(path),**{k:record[k] for k in ['cases','points','methods','reverse_order_exact']}))
        public=extracted_base/'PUBLIC_ENTRY_FRESH_ARCHIVE_NEXT9.json';assert read_json(public)['status']=='PASS'
        record=dict(status='PASS',timestamp=utc_now(),tag=tag,final_payload=final,archive_path=archive.relative_to(ROOT).as_posix(),
            archive_sha256=archive_sha,archive_bytes=archive.stat().st_size,archive_integrity=verification,
            scopes=scopes,total_synthetic_series_procedures=1680,total_scalar_observations=860160,
            models_per_observation=9,all_frozen_probabilities_bitwise=True,public_entry_receipt_sha256=digest(public),
            public_NEXT9_cases=168,canonical_public_cases=9,all_original_payload_bytes_unchanged=True,
            raw_data_retraining=False,original_workspace_fallback_denied=True,fresh_cache_relative_path='nc',
            imported_activity_provenance='Child ledger/runtimejobs identical hashes; only post-admission FINAL_AUDIT copied; parent wait not counted',
            receipt_external_to_this_archive=True)
        atomic(out/'REPRODUCTION_DONE.json',record)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--final',action='store_true');a=p.parse_args();main(a.tag,a.final)
