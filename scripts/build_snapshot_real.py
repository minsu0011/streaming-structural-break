"""Explicit real-handoff allowlist with selected models, hashes and verified ZIP."""
import argparse, hashlib, json, os, re, sys, zipfile
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.utils.artifacts import write_json, sha256, utc_now, commit_id

DIRECTORIES=('src','tests','scripts','docs','configs','vendor/official','checkpoints','artifacts/audits','artifacts/baselines','artifacts/features','artifacts/validation','artifacts/real','experiments/results')
TEXT_SUFFIXES={'.py','.json','.md','.csv','.txt','.toml','.xml','.ipynb'}
SELF_MANIFESTS={'SOURCE_MANIFEST.json','24_SOURCE_MANIFEST.json'}
EXCLUDED_PARTS={'.crunchdao','.crunch','.venv','.repro-venv','.figure-venv','.git','__pycache__','.pytest_cache','reduced_private_until_final','reproduction','logs'}


def included_files(provisional_candidates=()):
    files=set()
    for directory in DIRECTORIES:
        for path in (ROOT/directory).rglob('*'):
            relative=path.relative_to(ROOT)
            if path.is_file() and path.suffix in TEXT_SUFFIXES and path.name!='FINALIZATION_STATE.json' and not EXCLUDED_PARTS.intersection(relative.parts):files.add(path)
    files.update(p for p in ROOT.iterdir() if p.is_file() and (p.suffix in TEXT_SUFFIXES or p.name in ('.gitignore','.gitattributes')) and p.name not in SELF_MANIFESTS)
    for name in ('GROUP_ASSIGNMENTS.parquet','artifacts/validation/GROUP_ASSIGNMENTS.parquet','artifacts/audits/HISTORICAL_GROUPS.parquet'):
        if (ROOT/name).exists():files.add(ROOT/name)
    for suffix in ('png','pdf','svg'):
        figure=ROOT/'artifacts/real/figures'/f'DEV_RESULTS_OVERVIEW.{suffix}'
        if figure.exists():files.add(figure)
    frozen_path=ROOT/'DEV_CHAMPION_CANDIDATES.json';candidate_models={}
    if frozen_path.exists():
        frozen=json.loads(frozen_path.read_text())
        for selected in frozen['candidates'].values():
            candidate_models[selected['experiment_id']]=selected
    if provisional_candidates:
        state=json.loads((ROOT/'artifacts/real/RESEARCH_STATE.json').read_text());known={r['experiment_id']:r for r in state['results'] if r.get('status')=='COMPLETE' and r.get('candidate')}
        for eid in provisional_candidates:
            if eid not in known:raise ValueError('Only a completed known CV candidate can be packaged provisionally')
            candidate_models[eid]={'experiment_id':eid}
    for selected in candidate_models.values():
        if 'model_directory' in selected:
            for directory in (ROOT/selected['model_directory'],ROOT/'artifacts/models/real'/selected['experiment_id']):
                files.update(p for p in directory.glob('*') if p.is_file() and p.suffix in ('.joblib','.json'))
        else:files.update((ROOT/'artifacts/models/real'/selected['experiment_id']).glob('fold_*.joblib'))
        oof=ROOT/'data/processed/oof'/selected['experiment_id']/'MANIFEST.json'
        if oof.exists():
            # Copy provenance to a dedicated report path, without including OOF rows.
            target=ROOT/'artifacts/handoff/selected_oof_manifests'/selected['experiment_id']/'MANIFEST.json'
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(oof.read_bytes());files.add(target)
    for path in files:
        relative=path.relative_to(ROOT)
        if EXCLUDED_PARTS.intersection(relative.parts) or relative.parts[0]=='data' or path.name in ('token','crunch.json') or path.name.startswith('.env'):
            raise RuntimeError('Forbidden snapshot path: '+relative.as_posix())
    return sorted(files,key=lambda p:p.relative_to(ROOT).as_posix())


def executable_records(records):
    return {r['path']:r['sha256'] for r in records if r['path'].endswith(('.py','.toml')) or r['path'].startswith('configs/') or Path(r['path']).name in ('requirements.txt','requirements-lock.txt')}


def check_sensitive_content(files):
    # Known secrets remain in memory only. Neither values nor matching excerpts
    # are ever included in an exception, a log, or the archive manifest.
    known=[]
    token_path=ROOT/'.crunchdao/token'
    if token_path.is_file():
        value=token_path.read_bytes().strip()
        if len(value)>=12:known.append(value)
    api_key=os.environ.get('CRUNCH_API_KEY','').encode()
    if len(api_key)>=12:known.append(api_key)
    signed_url=re.compile(rb'https?://[^\s<>"\x27]+[?&](?:X-Amz-Signature|X-Goog-Signature|Signature|token|sig)=[A-Za-z0-9%+/=_-]{12,}',re.IGNORECASE)
    for path in files:
        content=path.read_bytes()
        if any(value in content for value in known) or signed_url.search(content):
            raise RuntimeError('Sensitive-content check failed for an allowlisted file; no matching value was logged')
    return {'status':'PASS','files_scanned':len(files),'known_secret_values_checked':len(known),
            'signed_url_pattern_checked':True,'scope':'Exact current local token/API-key bytes when present, plus signed-URL patterns. This is not a proof against every possible unknown secret.'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--desktop',action='store_true');parser.add_argument('--require-reproduction',action='store_true')
    parser.add_argument('--candidate',action='append',default=[],help='Include completed provisional CV models for an intermediate reproduction archive');args=parser.parse_args()
    if args.desktop and args.candidate:raise ValueError('Final Desktop ZIP contains only the frozen final selections')
    if args.desktop:
        audit=json.loads((ROOT/'artifacts/validation/FINAL_AUDIT_REAL.json').read_text())
        if audit['status']!='PASS':raise RuntimeError('Final desktop handoff requires a passing real-data audit')
    files=included_files(args.candidate);sensitive=check_sensitive_content(files)
    records=[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha256(p)} for p in files]
    reproduction_path=ROOT/'artifacts/validation/REPRODUCTION_REPORT.json'
    reproduction=json.loads(reproduction_path.read_text()) if reproduction_path.exists() else {}
    matches=reproduction.get('status')=='PASS' and reproduction.get('executable_sources_sha256')==executable_records(records)
    if args.require_reproduction and not matches:raise RuntimeError('Executable sources differ from the verified separate-environment reproduction')
    manifest={'timestamp':utc_now(),'git_commit':commit_id(),'files':records,'self_hash_excluded':True,
        'self_hash_exclusions':sorted(SELF_MANIFESTS),'reproduced_executable_sources_match':matches,'reproduction_archive':reproduction.get('archive'),
        'sensitive_content_check':sensitive,
        'scope':'Real DEV research code/config/tests/reports, frozen group assignment, selected fold models and final DEV model artifacts.',
        'excluded':['raw data','feature caches','per-row OOF arrays','all other model binaries','per-series target metadata','credentials','virtual environments','authenticated logs'],
        'LOCAL_FINAL_SEAL_OPENED':False}
    for name in SELF_MANIFESTS:write_json(ROOT/name,manifest)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    if args.desktop:
        import ctypes
        buffer=ctypes.create_unicode_buffer(32768)
        if ctypes.windll.shell32.SHGetFolderPathW(None,0x10,None,0,buffer)!=0:raise RuntimeError('Windows Desktop location could not be resolved')
        directory=Path(buffer.value)
        if not directory.is_dir():raise RuntimeError('Resolved Desktop directory does not exist')
        output=directory/f'Structural_Break_GPT_전달용_REAL_{stamp}.zip'
    else:output=ROOT/f'SRC_REAL_SNAPSHOT_{stamp}.zip'
    if output.exists():raise RuntimeError('Snapshot output already exists')
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for path in files+[ROOT/name for name in sorted(SELF_MANIFESTS)]:archive.write(path,path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:raise RuntimeError('ZIP CRC verification failed')
        if len(archive.namelist())!=len(records)+len(SELF_MANIFESTS):raise RuntimeError('ZIP inventory mismatch')
        for record in records:
            data=archive.read(record['path'])
            if len(data)!=record['bytes'] or hashlib.sha256(data).hexdigest()!=record['sha256']:raise RuntimeError('ZIP source hash mismatch')
    report={'status':'PASS','timestamp':utc_now(),'archive':str(output),'bytes':output.stat().st_size,'sha256':sha256(output),'files':len(records)+len(SELF_MANIFESTS),
        'raw_data_included':False,'credentials_included':False,'seal_target_metadata_included':False,'selected_models_included':(ROOT/'DEV_CHAMPION_CANDIDATES.json').exists(),
        'sensitive_content_check':sensitive,
        'reproduced_executable_sources_match':matches,'desktop_handoff':args.desktop,'provisional_cv_models_included':args.candidate}
    write_json(ROOT/'artifacts/handoff'/('DESKTOP_ZIP_VERIFICATION_REAL.json' if args.desktop else 'SNAPSHOT_VERIFICATION_REAL.json'),report)
    print(json.dumps(report,ensure_ascii=True),flush=True)


if __name__=='__main__':main()
