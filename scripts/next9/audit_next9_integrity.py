"""Recheck original protected bytes and retained dirty Git patches."""
from pathlib import Path
import sys
import argparse
import subprocess
import hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now

def main(stage):
    kind='FINAL_AUDIT' if stage=='FINAL' else 'SOURCE_INTEGRITY_RESEARCH'
    if stage=='FINAL':assert read_json(BASE/'runtime/FINAL_AUDIT_ADMISSION_NEXT9.json')['status']=='PASS'
    with activity(f'{stage} protected originals, canonical models and dirty Git integrity',kind,'B9_NORMALIZATION_STATE_AVAILABILITY'):
        protected=read_json(BASE/'PROTECTED_BEFORE_NEXT9_SHA256.json');mismatches=[];size=0
        for name,sha in protected.items():
            path=ROOT/name
            if not path.is_file():mismatches.append(dict(path=name,reason='MISSING'));continue
            size+=path.stat().st_size
            actual=digest(path)
            if actual!=sha:mismatches.append(dict(path=name,reason='HASH_CHANGED',expected=sha,actual=actual))
        patches={}
        for name,args in [('STAGED_PATCH_BEFORE.diff',['diff','--cached','--binary']),('WORKTREE_PATCH_BEFORE.diff',['diff','--binary'])]:
            result=subprocess.run(['git',*args],cwd=ROOT,capture_output=True,check=True)
            actual=hashlib.sha256(result.stdout).hexdigest();expected=digest(BASE/name)
            patches[name]=dict(expected_sha256=expected,current_sha256=actual,unchanged=actual==expected)
        record=dict(status='PASS' if not mismatches and all(v['unchanged'] for v in patches.values()) else 'FAIL',
            timestamp=utc_now(),stage=stage,protected_files=len(protected),protected_bytes=size,mismatches=mismatches,dirty_patches=patches,
            protected_manifest_sha256=digest(BASE/'PROTECTED_BEFORE_NEXT9_SHA256.json'),
            namespace_exception='Earlier default pytest updated old .pytest_cache/v/cache/nodeids; original prior bytes unavailable, not restored. Later pytest disables cacheprovider. This is not a claim of zero writes outsideNEXT9.',
            namespace_exception_receipt_sha256=digest(BASE/'repairs/PYTEST_CACHE_NAMESPACE_EXCEPTION_NEXT9.json'))
        atomic(BASE/'final_integrity'/f'{stage}_PROTECTED_NEXT9.json',record)
        assert record['status']=='PASS',mismatches

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=['PRE_FINAL','FINAL']);main(p.parse_args().stage)
