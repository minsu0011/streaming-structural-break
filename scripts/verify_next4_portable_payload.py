"""Replay inference from a copied raw-free source/model payload in fresh processes."""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next4.handoff_payload import collect


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    if not re.fullmatch('[A-Za-z0-9_]+',args.tag):raise ValueError('Safe immutable verification tag required')
    paths,files,excluded,missing=collect(require_final=False)
    out=BASE/'portable_verification'/args.tag;out.mkdir(parents=True,exist_ok=False)
    payload=out/'payload';payload.mkdir()
    for path in paths:
        relative=path.relative_to(ROOT);destination=(payload/relative).resolve()
        assert destination.is_relative_to(payload.resolve())
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,destination)
        assert digest(destination)==files[relative.as_posix()]['sha256']
    assert not (payload/'data').exists() and not list(payload.rglob('X_train.parquet'))
    atomic_json(out/'COPY_MANIFEST.json',{'created_utc':utc_now(),'files':files,
        'missing_final_reports_at_copy':missing,'excluded_reports':excluded},immutable=True)
    records=[];specs=[('BL80','scripts/verify_next4_cold_api.py',['--worker','BL80','--output'],BASE/'cold_api/BL80_1.json'),
        ('B2','scripts/verify_next4_b2_readonly.py',['--worker-output'],BASE/'cold_api_readonly_v2/B2_1.json')]
    selected=ROOT/'DEV_CANDIDATES_NEXT4.json'
    if selected.exists():
        role=read_json(selected)['roles'].get('NEXT4_ROBUSTNESS')
        if role:
            cid=role['experiment_id']
            specs.append((cid,'scripts/verify_next4_candidate_api.py',['--model',cid,'--cold-worker-output'],
                BASE/'finalist_validation'/cid/'COLD/cold_1.json'))
    for label,script,options,reference in specs:
        target=out/(label+'_cold.json')
        environment=dict(os.environ,PYTHONPATH=str(payload),PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        process=subprocess.run([sys.executable,'-B',str(payload/script),*options,str(target)],cwd=payload,
            env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),timeout=240)
        (out/(label+'.log')).write_text(process.stdout,encoding='utf-8');process.check_returncode()
        result=read_json(target);original=read_json(reference)
        assert result['status']=='PASS' and original['status']=='PASS'
        assert result['prediction_sha256']==original['prediction_sha256']
        records.append({'model':label,'fresh_copied_payload_process':True,
            'matching_existing_prediction_sha256':result['prediction_sha256'],
            'worker_result_sha256':digest(target),'reference_result_sha256':digest(reference),
            'source_script_sha256':digest(payload/script),'new_fits':0})
        progress('Raw-free copied-payload replay completed',label+' fresh-process cold read-only inference matched original fixed synthetic fingerprint',phase='FINAL_AUDIT',counts_as_research=False)
        print('NEXT4 RAW-FREE COPIED PAYLOAD',label,'PASS',flush=True)
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'records':records,
        'copied_files':len(files),'copy_manifest_sha256':digest(out/'COPY_MANIFEST.json'),
        'all_copied_file_sha256_verified':True,'raw_competition_data_present':False,
        'separate_source_and_model_directory':True,'fresh_Python_processes':len(specs),
        'original_pinned_interpreter_environment_reused':True,'new_environment_installation':False,
        'fresh_raw_training_reproduction':False,'new_fits':0,'real_or_seal_rows':0,
        'scope':'Moved source/model inference replay with synthetic fixed observations and original exact pinned environment. This is not an independent raw-to-training reproduction.',
        'source_sha256':digest(Path(__file__)),
        'files_sha256':{p.name:digest(p) for p in out.iterdir() if p.is_file()}},immutable=True)
    print('NEXT4 PORTABLE PAYLOAD VALIDATION COMPLETE',len(files),'copied files',flush=True)


if __name__=='__main__':main()
