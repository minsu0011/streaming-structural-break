"""Fresh final-audit byte, environment, guard and prior-work preservation checks."""
import argparse
import hashlib
import importlib.metadata
from pathlib import Path
import re
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next.guard import SealGuard,SealAccessError


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    if not re.fullmatch('[A-Za-z0-9_]+',args.tag):raise ValueError('Safe immutable final-audit tag required')
    session=read_json(BASE/'SESSION.json');assert session.get('audit_started_utc')
    out=BASE/'final_audit'/args.tag/'frozen_boundary';out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'START.json',{'created_utc':utc_now(),'audit_started_utc':session['audit_started_utc'],
        'source_sha256':digest(Path(__file__))},immutable=True)
    refs=read_json(BASE/'preflight/FROZEN_REFERENCES.json');guard=SealGuard(ROOT);guard.check_state()
    progress('Final frozen boundary audit','Raw byte hashes, unchanged locked partitions, sealed membership and original source/index preservation',phase='FINAL_AUDIT',counts_as_research=False)
    raw={name:digest(ROOT/'data/raw'/name) for name in refs['raw_sha256']};assert raw==refs['raw_sha256']
    assert guard.split.digest==refs['split_sha256'] and guard.lock_sha==refs['seal_lock_sha256']
    assert guard.ledger_sha==refs['reduced_ledger_sha256']
    assert len(guard.dev_ids)==8000 and len(guard.seal_ids)==2000 and not guard.dev_ids&guard.seal_ids
    blocked={}
    for purpose in ('training','validation','OOF','feature generation','label access'):
        try:guard.admit([min(guard.seal_ids)],purpose=purpose)
        except SealAccessError:blocked[purpose]=True
        else:raise AssertionError('Seal identity unexpectedly admitted')
    original=read_json(BASE/'preflight/PREEXISTING_SOURCE_SNAPSHOT.json')
    for name,record in original['files'].items():assert digest(ROOT/name)==record['sha256'],name
    assert len(original['files'])==908
    saved_patch=(BASE/'preflight/INITIAL_STAGED_PATCH.diff').read_bytes()
    staged_patch=subprocess.check_output(['git','diff','--cached','--binary'],cwd=ROOT,stderr=subprocess.PIPE)
    assert staged_patch==saved_patch
    for name,sha in refs['source_sha256'].items():assert digest(ROOT/name)==sha,name
    checked_models={}
    for name,role in refs['roles'].items():
        path=ROOT/role.get('model_file',role.get('model',''))
        expected=role.get('actual_model_sha256',role.get('sha256'))
        assert path.is_file() and digest(path)==expected
        checked_models[path.relative_to(ROOT).as_posix()]=expected
    pins={}
    for line in (ROOT/'requirements-lock.txt').read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.startswith('#'):continue
        name,expected=line.split('==');actual=importlib.metadata.version(name)
        assert actual==expected;pins[name]=actual
    assert len(pins)==63
    delivery=refs['NEXT3_20_file_delivery'];folder=Path(delivery['folder'])
    for name,record in delivery['files'].items():assert digest(folder/name)==record['sha256']
    assert len(list(folder.iterdir()))==20
    assert read_json(ROOT/'artifacts/next3/SESSION.json')['status']=='STOPPED_BY_USER'
    guard.check_state()
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'audit_started_utc':session['audit_started_utc'],
        'raw_byte_sha256':raw,'raw_hashing_scope':'Whole-file byte hashing only; no seal observation or target query.',
        'DEV_series':8000,'seal_series':2000,'DEV_seal_overlap':0,'blocked_seal_admissions':blocked,
        'split_sha256':guard.split.digest,'seal_lock_sha256':guard.lock_sha,'reduced_ledger_sha256':guard.ledger_sha,
        'all_seals_unopened':True,'new_reduced_usage':0,'new_public_or_cloud_submission_by_this_check':0,
        'preexisting_source_files_unchanged':908,'preexisting_snapshot_sha256':digest(BASE/'preflight/PREEXISTING_SOURCE_SNAPSHOT.json'),
        'staged_patch_byte_equal':True,'staged_patch_sha256':hashlib.sha256(staged_patch).hexdigest(),
        'NEXT3_user_stop_preserved':True,'prior_Desktop_20_files_unchanged':True,
        'frozen_model_sha256':checked_models,'exact_environment_pins':pins,'python':sys.version,
        'new_fits':0,'seal_rows':0,'source_sha256':digest(Path(__file__))},immutable=True)
    print('NEXT4 FINAL FROZEN BOUNDARY AUDIT PASS; 908 sources and 63 pins',flush=True)


if __name__=='__main__':main()
