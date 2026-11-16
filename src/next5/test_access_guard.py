"""Pytest-only denial of canonical raw data and old per-series array reads.

Legacy tests may load frozen model weights and use their existing synthetic
fixtures. They do not need canonical competition observations or old DEV OOF.
"""
import os
from pathlib import Path
from src.next5.artifacts import ROOT,atomic_json,utc_now

_originals=[]
_calls=[]
_blocked=[]


def classify_path(value):
    if not isinstance(value,(str,bytes,os.PathLike)):
        value=getattr(value,'name',None)
    if not isinstance(value,(str,bytes,os.PathLike)):return None
    try:path=Path(os.fsdecode(value)).resolve()
    except (ValueError,OSError):return None
    if path.is_relative_to(ROOT/'data'):return 'canonical_project_data_forbidden_in_unit_tests'
    if path.suffix.lower() in ('.npy','.npz','.parquet') and path.is_relative_to(ROOT/'artifacts') and not path.is_relative_to(ROOT/'artifacts/next5'):
        return 'historical_per_series_arrays_forbidden_in_NEXT5_tests'
    return None


def checked(function,name):
    def wrapper(source,*args,**kwargs):
        reason=classify_path(source)
        _calls.append(name)
        if reason:
            _blocked.append({'reader':name,'reason':reason,'path':str(source)})
            raise RuntimeError('NEXT5 full-test access guard: '+reason)
        return function(source,*args,**kwargs)
    return wrapper


def pytest_configure(config):
    import numpy as np
    import pandas as pd
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
    for module,name in ((np,'load'),(pd,'read_parquet'),(ds,'dataset'),(pq,'read_table')):
        function=getattr(module,name);_originals.append((module,name,function));setattr(module,name,checked(function,module.__name__+'.'+name))


def pytest_unconfigure(config):
    for module,name,function in reversed(_originals):setattr(module,name,function)
    directory=os.environ.get('NEXT5_TEST_AUDIT_DIRECTORY')
    if directory:
        path=Path(directory).resolve()
        if not path.is_relative_to(ROOT/'artifacts/next5/final_audit'):raise RuntimeError('Owned test audit path required')
        atomic_json(path/'ACCESS_GUARD.json',{'created_utc':utc_now(),'status':'PASS' if not _blocked else 'BLOCKED_READ_ATTEMPTS',
            'instrumented_read_calls':len(_calls),'blocked_reads':_blocked,
            'scope':'NumPy, pandas and PyArrow entry points during complete tests; synthetic temporary fixtures and immutable model weights remain allowed.',
            'not_an_operating_system_sandbox':True},immutable=True)
