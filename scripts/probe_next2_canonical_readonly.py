"""Cold-process filesystem audit for canonical compact and tree-used entries."""
import sys
sys.dont_write_bytecode=True
import os
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
role=sys.argv[1]
if role not in ('DISTILLED','PERFORMANCE'):
    raise ValueError('Explicit role required')
attempts=[]
null_device_opens=[]


def audit(event,args):
    write=False
    if event=='open':
        path,mode,flags=args
        if str(path).casefold()==os.devnull.casefold():
            null_device_opens.append(str(path))
            return
        write=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
    if event in ('os.mkdir','os.remove','os.rmdir','os.rename','os.chmod','os.truncate'):
        write=True
    if write:
        attempts.append(event)
        raise RuntimeError('Canonical read-only probe intercepted write: '+event)


sys.addaudithook(audit)
import importlib
entry=importlib.import_module('next2_canonical_'+role.lower()+'_main')
import numpy as np
from src.next.readonly_bootstrap import cache_requests_suppressed
h=(np.sin(np.arange(256)*.37)+.2*np.cos(np.arange(256)*.11)).astype(np.float32)
o=np.cos(np.arange(129)*.7).astype(np.float32)
runtime='tree_used_v1' if role=='DISTILLED' else 'compact_v1'
folder=ROOT/'artifacts/next2/canonical_deployment'/(role+'_'+runtime)
result=list(entry.infer([(h,iter(o))],folder))
assert result[0] is None and len(result)==len(o)+1
assert all(np.isfinite(p) and 0<=p<=1 for p in result[1:])
assert not attempts
print(json.dumps({'status':'PASS','role':role,'runtime':runtime,'filesystem_write_attempts':0,
    'cold_project_import_after_hook':True,'points':len(o),'first_none':True,
    'null_device_opens':len(null_device_opens),
    'suppressed_numba_disk_cache_requests':len(cache_requests_suppressed())}),flush=True)
