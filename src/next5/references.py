"""Load hash-bound matched ResearchPool OOF components and fixed combinations."""
import numpy as np
from src.next5.artifacts import BASE,read_json,digest
from src.next5.data import admitted_npz


def component(kind,scope,split,metadata,guard):
    directory=BASE/'references/fits'/split/scope/kind
    result=read_json(directory/'RESULTS.json')
    if result['status']!='COMPLETE' or digest(directory/'OOF.npz')!=result['OOF_sha256']:
        raise RuntimeError('Complete unchanged reference OOF required')
    values=admitted_npz(directory/'OOF.npz',guard,'Matched ResearchPool reference OOF')
    rows=values['global_row'];prediction=values['prediction']
    if len(np.unique(rows))!=len(rows) or not np.array_equal(metadata['dataset_id'][rows],values['dataset_id']):
        raise RuntimeError('Reference global row identity mismatch')
    if not np.isfinite(prediction).all():raise RuntimeError('Nonfinite reference OOF')
    full=np.full(len(metadata['dataset_id']),np.nan,dtype=np.float64);full[rows]=prediction
    return full,{'kind':kind,'scope':scope,'split':split,'OOF_sha256':result['OOF_sha256'],'result_sha256':digest(directory/'RESULTS.json')}


def primary(scope,split,metadata,guard):
    j,jr=component('J102',scope,split,metadata,guard);c,cr=component('C205',scope,split,metadata,guard)
    return .8*j+.2*c,c,[jr,cr]
