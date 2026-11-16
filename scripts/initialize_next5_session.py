"""Start an observed NEXT5 session, preserving prior work before data access."""
from pathlib import Path
import csv
import os
import platform
import shutil
import subprocess
import sys
import time
import psutil
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next5.artifacts import BASE, atomic_json, digest, read_json, utc_now, progress


def main():
    BASE.mkdir(parents=True, exist_ok=False)
    old4 = read_json(ROOT/'artifacts/next4/SESSION.json')
    old3 = read_json(ROOT/'artifacts/next3/SESSION.json')
    assert old4['status']=='COMPLETE' and old3['status']=='STOPPED_BY_USER'
    assert old3['automatic_resume_authorized'] is False
    stamp = utc_now()
    atomic_json(BASE/'SESSION.json', {
        'status': 'ACTIVE_RESEARCH', 'started_utc': stamp, 'monotonic_start': time.monotonic(),
        'pid_at_start': os.getpid(), 'target_productive_research_seconds': 36000,
        'minimum_audit_seconds': 3000, 'maximum_audit_seconds': 3600,
        'maximum_family_count': 7, 'maximum_screen_candidates': 35,
        'maximum_full_candidates': 20, 'maximum_shadow_finalists': 3,
        'shadow_opened': False, 'all_seals_unopened': True,
        'new_reduced_uses': 0, 'new_public_submissions': 0, 'new_cloud_submissions': 0,
        'modeling_blocked': True, 'modeling_block_reason': 'Initial preflight and prospective guards pending',
        'NEXT4_status_preserved': 'COMPLETE', 'NEXT3_status_preserved': 'STOPPED_BY_USER',
        'auto_resume_authorized': True,
        'timing_rule': 'Conservative observed CPU/log or substantive-agent activity; no backward credit, padding or early handoff. Shadow is outside the research minimum.'}, immutable=True)
    progress('NEXT5 initial preservation', 'Prior completed/stopped sessions, staged changes, source set and user specification', phase='PREFLIGHT')
    out = BASE/'preflight'; out.mkdir()
    attachment = Path('C:/Users/minsu/.codex/attachments/432a2eca-a67c-4a61-8186-241260e99b0a/pasted-text.txt')
    spec = ROOT/'docs/USER_SPECIFICATION_NEXT5.md'
    assert not spec.exists(); shutil.copyfile(attachment, spec)
    assert digest(attachment)==digest(spec)
    patch = subprocess.check_output(['git','diff','--cached','--binary'], cwd=ROOT, stderr=subprocess.PIPE)
    (out/'INITIAL_STAGED_PATCH.diff').write_bytes(patch)
    status = subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=normal'], cwd=ROOT)
    (out/'INITIAL_GIT_STATUS.txt').write_bytes(status)
    sources = set()
    for folder in ('src','scripts','tests','configs','docs'):
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and p.suffix in ('.py','.ps1','.json','.toml','.ini','.cfg','.md','.txt') and '__pycache__' not in p.parts:
                if 'next5' not in p.relative_to(ROOT).as_posix().lower(): sources.add(p)
    for p in ROOT.iterdir():
        if p.is_file() and p.suffix in ('.py','.json','.md','.csv','.txt','.toml') and 'NEXT5' not in p.name.upper(): sources.add(p)
    records = {p.relative_to(ROOT).as_posix(): {'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(sources)}
    atomic_json(out/'PREEXISTING_SOURCE_AND_REPORT_SNAPSHOT.json', {'created_utc':utc_now(),'files':records,
        'permitted_final_publication_exception': 'EXPERIMENT_LEDGER.csv may append NEXT5 after preserving exact prior rows and original bytes.'}, immutable=True)
    retained = out/'retained'; retained.mkdir()
    for name in ('EXPERIMENT_LEDGER.csv','DEV_CANDIDATES_NEXT4.json','FINAL_AUDIT_NEXT4.json','SOURCE_MANIFEST_NEXT4.json'):
        shutil.copyfile(ROOT/name, retained/name)
    atomic_json(out/'INITIAL_CONTEXT.json', {
        'status':'PRESERVED_PREFLIGHT_PENDING','created_utc':utc_now(),'specification_sha256':digest(spec),
        'staged_patch_sha256':digest(out/'INITIAL_STAGED_PATCH.diff'),'staged_patch_bytes':len(patch),
        'preexisting_files':len(records),'snapshot_sha256':digest(out/'PREEXISTING_SOURCE_AND_REPORT_SNAPSHOT.json'),
        'prior_session_sha256':{f'artifacts/{stage}/SESSION.json':digest(ROOT/f'artifacts/{stage}/SESSION.json') for stage in ('next3','next4')},
        'python':sys.version,'platform':platform.platform(),'physical_RAM_bytes':psutil.virtual_memory().total,
        'available_RAM_bytes':psutil.virtual_memory().available,'disk_free_bytes':psutil.disk_usage(str(ROOT)).free,
        'new_fits':0,'raw_or_shadow_rows_loaded':0}, immutable=True)
    print('NEXT5 SESSION INITIALIZED', stamp, 'preserved files', len(records), flush=True)


if __name__=='__main__': main()
