"""Guarded full OOF access for new, immutable NEXT2 experiments."""
from pathlib import Path
import json
import numpy as np
from src.next2.io import validate,is_next2
from src.next.candidate_io import load_full_oof as old_load_full_oof,validate_sources
from src.scoring.ts_auc import ts_auc
from src.utils.artifacts import sha256


def load_full_oof(data,eid):
    root=data.guard.root
    out=root/'artifacts/next2/experiments'/eid
    if not (out/'RESULTS.json').exists():
        return old_load_full_oof(data,eid)
    report=json.loads((out/'RESULTS.json').read_text(encoding='utf-8'))
    candidate=report['candidate']
    if report['status']=='BUG' or report.get('full',{}).get('status')!='COMPLETE':
        raise ValueError('Complete valid five-fold OOF required')
    if candidate.get('parity_reference'):
        return old_load_full_oof(data,candidate['parity_reference'])
    validate(candidate)
    if not is_next2(candidate):
        validate_sources(candidate)
    if candidate.get('blend'):
        from src.next2.blend import source_hash
        if candidate['blend']['implementation_sha256']!=source_hash():
            raise RuntimeError('Fixed OOF blend implementation changed')
    policy=json.loads((out/'full/POLICY.json').read_text(encoding='utf-8'))
    if policy['candidate']!=candidate or policy['policy_sha256']!=sha256(root/'configs/next2/RESEARCH_POLICY_LOCK.json'):
        raise RuntimeError('NEXT2 OOF nomination identity changed')
    metadata=data.rows()
    prediction=np.full(len(metadata['target']),np.nan,dtype=np.float32)
    sources={'policy_sha256':sha256(out/'full/POLICY.json')}
    for fold in range(5):
        directory=out/'full'
        receipt=json.loads((directory/f'fold_{fold}.json').read_text(encoding='utf-8'))
        if receipt['status']!='PASS':
            raise RuntimeError('Invalid NEXT2 fold status')
        for name,digest in receipt['files_sha256'].items():
            if sha256(directory/name)!=digest:
                raise RuntimeError('NEXT2 OOF artifacts changed')
            sources[name]=digest
        rows=np.load(directory/f'fold_{fold}_global_rows.npy',allow_pickle=False)
        np.testing.assert_array_equal(rows,np.flatnonzero(metadata['fold']==fold))
        values=np.load(directory/f'fold_{fold}.npy',allow_pickle=False)
        assert values.dtype==np.float32 and np.isfinite(values).all()
        prediction[rows]=values
        score=ts_auc(metadata['target'][rows],values,metadata['time_online'][rows])
        np.testing.assert_allclose(score,report['full']['fold_scores'][fold],rtol=0,atol=1e-14)
    assert np.isfinite(prediction).all()
    return candidate,report,metadata,prediction,sources
