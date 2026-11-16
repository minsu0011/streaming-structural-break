"""Explicit hash-bound read extension for the two existing RP BL80 caches."""
from pathlib import Path
import os
import sys
import hashlib
import numpy as np
from src.next9.common import BASE,ROOT,read_json,digest


def install_reference_boundary():
    receipt=read_json(BASE/'REFERENCE_CACHE_ACCESS_NEXT9.json')
    if receipt['status']!='PASS':raise ValueError('RP reference cache admission required')
    permitted={str((ROOT/p).resolve()).lower() for p in receipt['files_sha256']}
    writing_roots=[(ROOT/p).resolve() for p in ['src/next9','scripts/next9','artifacts/next9']]
    rp=(ROOT/'data/next5/research_5d54e316ff5bf25d').resolve();violations=[]
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve();mode=args[1] if len(args)>1 else None;flags=args[2] if len(args)>2 else 0
        writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)))
        reason=None
        if writing and p.is_relative_to(ROOT) and not any(p.is_relative_to(r) for r in writing_roots):reason='Legacy write forbidden'
        if not writing and p.is_relative_to(ROOT/'data') and not p.is_relative_to(rp) and str(p).lower() not in permitted:reason='Unadmitted data value path'
        if reason:
            violations.append(dict(path=str(p),reason=reason));raise PermissionError(reason)
    sys.addaudithook(audit)
    return violations


def matrix(kind,bank):
    if kind not in ['J102','C205']:raise ValueError('Only frozen BL80 components')
    receipt=read_json(BASE/'REFERENCE_CACHE_ACCESS_NEXT9.json')
    record=receipt['caches'][kind]
    for p in [record['manifest'],record['features']]:
        if digest(ROOT/p)!=receipt['files_sha256'][p]:raise ValueError('Reference cache hash changed')
    m=read_json(ROOT/record['manifest'])
    if m['identity']['row_ids_sha256']!=hashlib.sha256(bank.row_ids.tobytes()).hexdigest():
        raise ValueError('Exact RP reference row ordering mismatch')
    if m['identity']['split_sha256']!=bank.guard.split.digest or m['series']!=6400:
        raise ValueError('Original RP membership mismatch')
    x=np.load(ROOT/record['features'],mmap_mode='r',allow_pickle=False)
    if x.shape[0]!=len(bank.row_ids) or x.shape[1]!=len(m['feature_names']):
        raise ValueError('Reference feature schema mismatch')
    return x
