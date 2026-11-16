"""Reuse original exact J102/C205 OOF without fitting or loading their estimators."""
import numpy as np
from src.next4.artifacts import ROOT,digest,read_json
from src.next2.blend import combine


def fold_predictions(data,fold,scope,expected_rows):
    if scope not in ('screen','full') or fold not in range(5):raise ValueError('Frozen scope and fold required')
    values={};sources={}
    for name,namespace,eid in [('J102','next','J102_SERIAL_VARIANCE_512'),('C205','next2','C205_DEPENDENCE')]:
        directory=ROOT/'artifacts'/namespace/'experiments'/eid/scope
        receipt=read_json(directory/f'fold_{fold}.json')
        if receipt['status']!='PASS':raise RuntimeError('Frozen OOF receipt is not successful')
        for filename in (f'fold_{fold}.npy',f'fold_{fold}_global_rows.npy'):
            if digest(directory/filename)!=receipt['files_sha256'][filename]:raise RuntimeError('Frozen OOF bytes changed')
            sources[(directory/filename).relative_to(ROOT).as_posix()]=digest(directory/filename)
        rows=np.load(directory/f'fold_{fold}_global_rows.npy',allow_pickle=False)
        np.testing.assert_array_equal(rows,expected_rows)
        prediction=np.load(directory/f'fold_{fold}.npy',allow_pickle=False)
        if prediction.dtype!=np.float32 or not np.isfinite(prediction).all():raise ValueError('Finite original float32 OOF required')
        values[name]=prediction
    values['BL80']=combine(values['J102'],values['C205'],.8)
    data.guard.check_state()
    return values,sources
