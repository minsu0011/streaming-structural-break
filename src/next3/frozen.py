"""Read and validate immutable prior models, coordinates and feature contracts."""
from pathlib import Path
import numpy as np
from src.next3.artifacts import ROOT, read_json, digest
from src.next2.io import design, model_class, validate
from src.next.candidate_io import validate_sources

J102 = 'J102_SERIAL_VARIANCE_512'
C205 = 'C205_DEPENDENCE'
BL80 = 'BL2_C205_DEPENDENCE_80'


def candidate(eid):
    namespace = 'next' if eid == J102 else 'next2'
    directory = ROOT/'artifacts'/namespace/'experiments'/eid
    report = read_json(directory/'RESULTS.json')
    spec = report['candidate']
    validate(spec)
    if eid == J102:
        validate_sources(spec)
    return spec, directory


def fold_artifact(data, eid, fold, scope):
    spec, base = candidate(eid)
    directory = base/scope
    report = read_json(directory/f'fold_{fold}.json')
    if report['status'] != 'PASS':
        raise RuntimeError('Frozen fold receipt failed')
    for name, expected in report['files_sha256'].items():
        if digest(directory/name) != expected:
            raise RuntimeError('Frozen fold bytes changed: '+name)
    global_rows = np.load(directory/f'fold_{fold}_global_rows.npy', allow_pickle=False)
    metadata = data.rows()
    policy = read_json(ROOT/'configs/next3/RESEARCH_POLICY_LOCK.json')
    allowed = np.ones(len(metadata['target']),dtype=bool)
    if scope == 'screen':
        allowed = np.isin(metadata['dataset_id'],policy['screen']['ids'])
    np.testing.assert_array_equal(global_rows,np.flatnonzero(allowed & (metadata['fold']==fold)))
    data.guard.partition(np.unique(metadata['dataset_id'][allowed & (metadata['fold']!=fold)]),
                         np.unique(metadata['dataset_id'][global_rows]))
    prediction = np.load(directory/f'fold_{fold}.npy', allow_pickle=False)
    model = model_class(spec).load(directory/f'fold_{fold}.joblib')
    return model, prediction, global_rows, report['files_sha256']
