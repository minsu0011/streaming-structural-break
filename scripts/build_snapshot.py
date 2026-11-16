"""Allowlisted, hash-manifested source snapshot; excludes raw data and credentials."""
import json
import sys
import zipfile
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.utils.artifacts import sha256,write_json,commit_id,utc_now

DIRECTORIES=('src','tests','scripts','docs','configs','vendor/official','checkpoints','artifacts/audits','artifacts/baselines','artifacts/features','artifacts/validation','experiments/results')
SUFFIXES={'.py','.json','.md','.csv','.txt','.toml','.xml','.ipynb'}

def source_files():
    files=set()
    for name in DIRECTORIES:
        for path in (ROOT/name).rglob('*'):
            if path.is_file() and path.suffix in SUFFIXES and '__pycache__' not in path.parts and path.stat().st_size<5_000_000:
                files.add(path)
    files.update(p for p in ROOT.iterdir() if p.is_file() and (p.suffix in SUFFIXES or p.name in ('.gitignore','.gitattributes')) and p.name!='24_SOURCE_MANIFEST.json')
    return sorted(files,key=lambda p:p.relative_to(ROOT).as_posix())

def main():
    raw=ROOT/'artifacts/audits/RAW_DATA_MANIFEST.json'
    if raw.exists() and json.loads(raw.read_text(encoding='utf-8')).get('status')=='ACQUIRED':
        from scripts.build_snapshot_real import main as real_main
        return real_main()
    files=source_files()
    records=[]
    for path in files:
        relative=path.relative_to(ROOT).as_posix()
        if any(part in ('.crunchdao','.venv','.repro-venv','data') for part in Path(relative).parts[:-1]) and not relative.startswith('src/data/'):
            raise RuntimeError('Forbidden snapshot path: '+relative)
        records.append({'path':relative,'bytes':path.stat().st_size,'sha256':sha256(path)})
    reproduction_path=ROOT/'artifacts/validation/REPRODUCTION_REPORT.json'
    reproduction=json.loads(reproduction_path.read_text(encoding='utf-8')) if reproduction_path.exists() else {}
    reproduced=reproduction.get('executable_sources_sha256',{})
    current_executable={r['path'] for r in records if r['path'].endswith(('.py','.toml')) or r['path'].startswith('configs/') or Path(r['path']).name in ('requirements.txt','requirements-lock.txt')}
    matches=bool(reproduced) and set(reproduced)==current_executable and all((ROOT/name).exists() and sha256(ROOT/name)==digest for name,digest in reproduced.items())
    manifest={'timestamp':utc_now(),'git_commit':commit_id(),'files':records,'reproduced_executable_sources_match':matches,'reproduction_archive':reproduction.get('archive'),'excluded':['raw data','parquet caches','model binaries','credentials','virtual environments','full logs'],
        'self_hash_excluded':True,'scope':'source/config/tests/small reports; blocked-auth research handoff'}
    write_json(ROOT/'24_SOURCE_MANIFEST.json',manifest)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    path=ROOT/f'SRC_SNAPSHOT_{stamp}.zip'
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for file in files+[ROOT/'24_SOURCE_MANIFEST.json']:
            archive.write(file,file.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        for record in records:
            import hashlib
            assert hashlib.sha256(archive.read(record['path'])).hexdigest()==record['sha256']
    report={'status':'PASS','archive':path.name,'bytes':path.stat().st_size,'sha256':sha256(path),'files':len(records)+1,'raw_data_included':False,'credentials_included':False,'reproduced_executable_sources_match':matches}
    write_json(ROOT/'artifacts/handoff/SNAPSHOT_VERIFICATION.json',report)
    print(json.dumps(report))

if __name__=='__main__':main()
