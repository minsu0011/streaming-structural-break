"""Verify snapshot hashes, extract into a new folder, and test in another venv."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.utils.artifacts import write_json,utc_now,sha256

def main():
    parser=argparse.ArgumentParser();parser.add_argument('archive');parser.add_argument('--python',default=str(ROOT/'.repro-venv/Scripts/python.exe'))
    parser.add_argument('--artifact-audit',choices=('prepared','final'),help='Additionally audit packaged artifacts using a checked junction to canonical data/raw; no target evaluation')
    args=parser.parse_args()
    archive_path=Path(args.archive).resolve();destination=(ROOT/'artifacts/reproduction'/archive_path.stem).resolve()
    destination.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(archive_path) as archive:
        manifest_bytes=archive.read('24_SOURCE_MANIFEST.json');manifest=json.loads(manifest_bytes)
        for record in manifest['files']:
            if hashlib.sha256(archive.read(record['path'])).hexdigest()!=record['sha256']:raise RuntimeError('Source hash mismatch')
        for name in archive.namelist():
            resolved=(destination/name).resolve()
            if not resolved.is_relative_to(destination):raise RuntimeError('Archive path escapes destination')
        archive.extractall(destination)
    interpreter=str(Path(args.python).resolve())
    env=os.environ.copy();env.pop('PYTHONPATH',None)
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):env[name]='1'
    commands=[['-m','pip','check'],['-m','pytest','-q','--junitxml=artifacts/validation/reproduction_pytest.xml'],['scripts/verify_scorer.py']]
    if args.artifact_audit:
        raw=(ROOT/'data/raw').resolve(strict=True);link=destination/'data/raw'
        if not raw.is_relative_to(ROOT.resolve()) or not link.resolve().is_relative_to(destination):
            raise RuntimeError('Raw-reference paths must remain in the intended workspace')
        if link.exists():raise RuntimeError('Extracted archive unexpectedly already contains a raw directory')
        link.parent.mkdir(parents=True,exist_ok=True)
        if os.name=='nt':
            junction_env=env.copy();junction_env['SB_REPRO_RAW_LINK']=str(link);junction_env['SB_REPRO_CANONICAL_RAW']=str(raw)
            junction=subprocess.run(['powershell','-NoProfile','-NonInteractive','-Command',
                "$ErrorActionPreference='Stop'; New-Item -ItemType Junction -Path $env:SB_REPRO_RAW_LINK -Target $env:SB_REPRO_CANONICAL_RAW | Out-Null"],
                env=junction_env,capture_output=True,text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            if junction.returncode:raise RuntimeError('Workspace-contained raw junction creation failed')
        else:link.symlink_to(raw,target_is_directory=True)
        if link.resolve()!=raw:raise RuntimeError('Resolved raw reference differs from canonical data/raw')
        audit=['scripts/final_audit_real.py']
        if args.artifact_audit=='prepared':audit+=['--prepared','--tests','artifacts/validation/pytest_real_checkpoint_13.xml']
        commands.append(audit)
    results=[]
    for command in commands:
        output=subprocess.run([interpreter,*command],cwd=destination,env=env,capture_output=True,text=True,timeout=180)
        log=destination/'artifacts/validation'/f'reproduction_{len(results)}.log';log.write_text(output.stdout+output.stderr,encoding='utf-8')
        results.append({'command':[interpreter,*command],'exit_code':output.returncode,'output_tail':output.stdout[-1000:],'log':str(log.relative_to(ROOT))})
    executable_sources={r['path']:r['sha256'] for r in manifest['files'] if r['path'].endswith(('.py','.toml')) or r['path'].startswith('configs/') or Path(r['path']).name in ('requirements.txt','requirements-lock.txt')}
    report={'status':'PASS' if all(r['exit_code']==0 for r in results) else 'FAIL','timestamp':utc_now(),'archive':archive_path.name,
        'archive_sha256':sha256(archive_path),'archive_bytes':archive_path.stat().st_size,'source_manifest_sha256':hashlib.sha256(manifest_bytes).hexdigest(),
        'source_hashes_verified':True,'separate_source_copy':True,'separate_virtual_environment':True,'python':interpreter,'commands':results,'executable_sources_sha256':executable_sources}
    report['artifact_audit_mode']=args.artifact_audit
    report['artifact_audit_raw_scope']='Checked workspace-contained reference to canonical raw; audit hashes raw bytes but does not evaluate target rows, retrain, or call the authenticated tester.' if args.artifact_audit else None
    write_json(ROOT/'artifacts/validation/REPRODUCTION_REPORT.json',report);print(json.dumps(report,indent=2))
    if report['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':main()
