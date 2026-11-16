"""Fresh-process audit hook forbids filesystem writes during cold import/infer."""
import sys
sys.dont_write_bytecode=True
import os
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
role=sys.argv[1]
if role not in ('DISTILLED','PERFORMANCE'):
    raise ValueError('Explicit research role required')
attempts=[]


def audit(event,args):
    write=False
    if event=='open':
        path,mode,flags=args
        write=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
    if event in ('os.mkdir','os.remove','os.rmdir','os.rename','os.chmod','os.truncate'):
        write=True
    if write:
        attempts.append(event)
        raise RuntimeError('Read-only probe intercepted a filesystem write: '+event)


sys.addaudithook(audit)
import importlib
entry=importlib.import_module('next2_'+role.lower()+'_main')
import numpy as np
from src.next.readonly_bootstrap import cache_requests_suppressed
index=np.arange(256)
h=(np.sin(index*.37)+.2*np.cos(index*.11)).astype(np.float32)
o=np.cos(np.arange(129)*.7).astype(np.float32)
result=list(entry.infer([(h,iter(o))],ROOT/'artifacts/next2/deployment_research'/role))
assert result[0] is None and len(result)==len(o)+1
assert all(np.isfinite(p) and 0<=p<=1 for p in result[1:])
assert not attempts
print(json.dumps({'status':'PASS','role':role,'filesystem_write_attempts':len(attempts),
    'cold_project_import_after_hook':True,'points':len(o),'first_none':True,
    'suppressed_numba_disk_cache_requests':len(cache_requests_suppressed())}),flush=True)
