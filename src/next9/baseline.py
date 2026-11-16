"""Exact excluded-group BL80 sources; no full-DEV deployment substitution."""
import numpy as np
from src.next9.common import BASE,ROOT,read_json,digest


def prior_base():
    return ROOT/read_json(BASE/'SESSION_NEXT9.json')['previous_NEXT8_base']


def _load(path,sha,bank,admitted):
    if digest(path)!=sha:raise ValueError('Frozen BL80 prediction hash mismatch')
    with np.load(path,allow_pickle=False) as z:
        rows=z['global_row'];ids=z['dataset_id']
        if not set(map(int,ids))<=set(admitted):raise ValueError('BL80 prediction membership mismatch')
        np.testing.assert_array_equal(bank.row_ids[rows],ids)
        p=z['prediction'].astype(np.float64)
    if p.shape!=rows.shape or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Aligned finite excluded BL80 probabilities required')
    if np.any(np.diff(rows)<=0):raise ValueError('Sorted unique excluded prediction rows required')
    return rows,p


def baseline_for(bank,scope,fold,requested,*,early=False,inner_oof=False):
    if early and inner_oof:
        raise ValueError('Early inner BL80 requires separately scoped t<=199 refits; normal OOF is forbidden')
    old=prior_base();sources=[]
    if early:
        directory=old/'early_baseline'/f'outer_{fold}_EARLY_T199'
        receipt=read_json(directory/'DONE.json')
        if receipt['scope']['training_ids_sha256']!=scope.provenance()['training_ids_sha256']:
            raise ValueError('Early BL80 excluded TRAIN mismatch')
        path=directory/'BL80_PREDICTIONS.npz'
        rows,p=_load(path,receipt['prediction_sha256'],bank,scope.validation_ids)
        sources.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=receipt['prediction_sha256']))
        if np.any(bank.times[requested]<400):raise ValueError('Original early outer reference only covers late stress')
    else:
        audit=read_json(old/'BASELINE_PROVENANCE_NEXT8.json')
        if audit['status']!='PASS':raise ValueError('Restored baseline provenance must pass')
        if inner_oof:
            path=ROOT/f'artifacts/next7/nested_base/fold_{fold}/BL80_INNER_OOF.npz'
            sha=audit['files_sha256'][path.relative_to(ROOT).as_posix()]
            rows,p=_load(path,sha,bank,scope.training_ids)
            sources.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=sha))
        else:
            components=[]
            for kind in ['J102','C205']:
                path=ROOT/f'artifacts/next5/references/fits/PRIMARY/full/{kind}/fold_{fold}/predictions.npz'
                sha=audit['files_sha256'][path.relative_to(ROOT).as_posix()]
                rr,pp=_load(path,sha,bank,scope.validation_ids)
                if components:np.testing.assert_array_equal(rr,rows)
                rows=rr;components.append(pp)
                sources.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=sha))
            p=.8*components[0]+.2*components[1]
    loc=np.searchsorted(rows,requested)
    if np.any(loc>=len(rows)) or not np.array_equal(rows[loc],requested):
        raise ValueError('Every requested row must have exact matched excluded BL80')
    return p[loc],dict(sources=sources,early=early,inner_oof=inner_oof,
        canonical_full_DEV_model_used=False,scope=scope.provenance())
