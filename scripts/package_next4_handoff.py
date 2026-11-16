"""Duration-gated complete source/evidence ZIP plus exactly twenty GPT summary files."""
from datetime import datetime,timezone
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next4.finalization import require_handoff_time
from src.next4.handoff_payload import collect
from src.next.guard import SealGuard

TWENTY='''00_GPT_READ_FIRST_NEXT4.md 01_GPT_HANDOFF_NEXT4.md MODEL_RESULTS_NEXT4.csv
CV_RESULTS_NEXT4.csv PAIRED_BOOTSTRAP_NEXT4.csv ALT_RESULTS_NEXT4.csv MECHANISM_RESULTS_NEXT4.csv
TIME_FRACTURE_NEXT4.csv COUNTEREXAMPLE_RESULTS_NEXT4.csv LONG_NULL_RESULTS_NEXT4.csv LATENCY_NEXT4.json
LEAKAGE_AUDIT_NEXT4.json PREFIX_INVARIANCE_NEXT4.json DETERMINISM_NEXT4.json REPRODUCTION_NEXT4.json
FINAL_AUDIT_NEXT4.json DEV_CANDIDATES_NEXT4.json BOTTLENECK_HISTORY_NEXT4.csv LIMITATIONS_NEXT4.md
NEXT_SESSION_PLAN_NEXT4.md'''.split()
assert len(TWENTY)==len(set(TWENTY))==20


def write_verified_archive(destination,members):
    destination=destination.resolve();temporary=destination.with_suffix(destination.suffix+'.partial')
    if destination.exists() or temporary.exists():raise FileExistsError('Never overwrite a prior delivery')
    if destination.parent!=temporary.parent:raise RuntimeError('Archive temporary path escaped destination')
    names=[name for _,name,_ in members]
    assert len(names)==len(set(names))
    for path,name,expected in members:
        assert '\\' not in name and not Path(name).is_absolute() and '..' not in Path(name).parts
        assert digest(path)==expected
    with temporary.open('xb') as stream:
        with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as archive:
            for path,name,_ in members:archive.write(path,name)
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None
        assert archive.namelist()==names
        for _,name,expected in members:
            if hashlib.sha256(archive.read(name)).hexdigest()!=expected:raise RuntimeError('Archived member hash mismatch: '+name)
    temporary.rename(destination)
    return {'path':str(destination),'bytes':destination.stat().st_size,'sha256':digest(destination),
        'entries':len(members),'CRC_verified':True,'all_member_SHA256_verified':True}


def main():
    timing=require_handoff_time();guard=SealGuard(ROOT);guard.check_state()
    audit=read_json(ROOT/'FINAL_AUDIT_NEXT4.json');selection=read_json(ROOT/'DEV_CANDIDATES_NEXT4.json')
    assert audit['status']=='PASS' and audit['checks'] and all(v is True for v in audit['checks'].values())
    assert selection['status']=='FROZEN_NEXT4_DEV_SELECTION'
    for name,sha in audit['evidence_sha256'].items():assert digest(ROOT/name)==sha
    progress('Verified GPT delivery packaging','Final timing/guard/audit gates passed; writing raw-free Desktop deliverables',phase='DELIVERY',counts_as_research=False)
    paths,files,excluded,missing=collect(require_final=True);assert not missing
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=BASE/'handoff'/stamp;out.mkdir(parents=True,exist_ok=False)
    manifest={'status':'FROZEN_NEXT4_PACKAGE_SOURCE_MANIFEST','created_utc':utc_now(),'files':files,
        'files_excluding_this_manifest':len(files),'uncompressed_bytes':sum(r['bytes'] for r in files.values()),
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'final_audit_sha256':digest(ROOT/'FINAL_AUDIT_NEXT4.json'),'selection_sha256':digest(ROOT/'DEV_CANDIDATES_NEXT4.json'),
        'observed_time_before_packaging':timing,'excluded_reports':excluded,
        'exclusions':['Raw competition observations/labels and historical raw-record manifests',
            'Real point-level OOF, features and training data caches','Credentials, environments and Git',
            'Prior ZIPs, repeated payload/workspace snapshots and obsolete maintenance copies',
            'Live supervisor/accounting/queue state; final observed timing is recorded in audit evidence'],
        'fitted_models':'Preserved research fit artifacts and frozen references, not additional selected deployment roles.',
        'small_array_scope':'Aggregate paired score draws only; no real point-level OOF arrays.',
        'raw_free_use':'Canonical and diagnostic inference can be replayed without competition data. Re-training requires separately supplied hash-matched raw files and regeneration of excluded DEV caches.',
        'self_hash':'The complete archive and this manifest are hashed in the external verification receipt; no recursive self hash.'}
    atomic_json(out/'SOURCE_MANIFEST_NEXT4.json',manifest,immutable=True)
    atomic_json(ROOT/'SOURCE_MANIFEST_NEXT4.json',manifest)
    manifest_path=ROOT/'SOURCE_MANIFEST_NEXT4.json'
    members=[(p,p.relative_to(ROOT).as_posix(),files[p.relative_to(ROOT).as_posix()]['sha256']) for p in paths]
    members.append((manifest_path,manifest_path.name,digest(manifest_path)))
    desktop=Path(subprocess.check_output(['powershell.exe','-NoProfile','-NonInteractive','-Command',
        "[Environment]::GetFolderPath('Desktop')"],text=True).strip()).resolve()
    assert desktop.is_dir()
    archive=desktop/('STRUCTURAL_BREAK_GPT_HANDOFF_NEXT4_'+stamp+'.zip')
    assert archive.resolve().parent==desktop
    full=write_verified_archive(archive,members)
    folder=desktop/('Structural_Break_NEXT4_과정과결과_20개_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    assert folder.resolve().parent==desktop
    folder.mkdir(exist_ok=False);twenty={}
    for name in TWENTY:
        source=ROOT/name;destination=folder/name;shutil.copyfile(source,destination)
        actual=digest(destination);assert actual==digest(source)==files[name]['sha256']
        twenty[name]={'sha256':actual,'bytes':destination.stat().st_size}
    assert len(list(folder.iterdir()))==20 and all(p.is_file() for p in folder.iterdir())
    summary_zip=write_verified_archive(folder.with_suffix('.zip'),[(folder/n,n,twenty[n]['sha256']) for n in TWENTY])
    guard.check_state()
    receipt={'status':'PASS','created_utc':utc_now(),'full_project_archive':full,
        'twenty_file_folder':str(folder),'twenty_file_count':20,'twenty_files':twenty,
        'twenty_file_archive':summary_zip,'manifest_sha256':digest(manifest_path),
        'raw_competition_data_included':False,'real_point_OOF_included':False,'credentials_included':False,
        'all_seals_unopened':True,'new_reduced_usage':0,'new_public_submission':0,'new_cloud_submission':0,
        'observed_time_before_packaging':timing,'observed_time_after_packaging':require_handoff_time(),
        'source_sha256':digest(Path(__file__))}
    atomic_json(out/'DESKTOP_DELIVERY_VERIFICATION_NEXT4.json',receipt,immutable=True)
    atomic_json(BASE/'handoff/DESKTOP_DELIVERY_POINTER.json',{'created_utc':utc_now(),
        'receipt':(out/'DESKTOP_DELIVERY_VERIFICATION_NEXT4.json').relative_to(ROOT).as_posix(),
        'sha256':digest(out/'DESKTOP_DELIVERY_VERIFICATION_NEXT4.json')})
    print('NEXT4 DESKTOP DELIVERY VERIFIED',full['entries'],'full-project files; exactly 20 summary files',flush=True)
    # Avoid locale-dependent console encoding of the Korean Desktop folder name.
    print('NEXT4 FULL ZIP',archive.name,full['sha256'],flush=True)


if __name__=='__main__':main()
