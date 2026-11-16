"""Original grouped OOF access for completed NEXT3 candidates."""
import numpy as np
from src.next3.artifacts import ROOT,BASE,read_json,digest
from src.next2.evaluation import load_full_oof as old_load_full_oof
from src.scoring.ts_auc import ts_auc


def load_full_oof(data, name):
    out=BASE/'experiments'/name/'full'
    if not (out/'RESULTS.json').exists():
        return old_load_full_oof(data,name)
    result=read_json(out/'RESULTS.json')
    if result['status']!='COMPLETE' or result['scope']!='full':
        raise RuntimeError('Complete full NEXT3 OOF required')
    metadata=data.rows()
    prediction=np.full(len(metadata['target']),np.nan,dtype=np.float32)
    sources={'result_sha256':digest(out/'RESULTS.json'),'policy_sha256':digest(out/'POLICY.json')}
    for fold in range(5):
        report=read_json(out/f'fold_{fold}.json')
        if report['status']!='PASS':raise RuntimeError('Failed NEXT3 fold receipt')
        for filename,expected in report['files_sha256'].items():
            if digest(out/filename)!=expected:raise RuntimeError('NEXT3 OOF file changed: '+filename)
            sources[filename]=expected
        rows=np.load(out/f'fold_{fold}_global_rows.npy',allow_pickle=False)
        values=np.load(out/f'fold_{fold}.npy',allow_pickle=False)
        np.testing.assert_array_equal(rows,np.flatnonzero(metadata['fold']==fold))
        data.guard.record_rows(metadata['dataset_id'][rows],purpose='NEXT3 original OOF diagnostic')
        assert values.dtype==np.float32 and np.isfinite(values).all()
        prediction[rows]=values
        score=ts_auc(metadata['target'][rows],values,metadata['time_online'][rows])
        np.testing.assert_allclose(score,result['fold_scores'][fold],rtol=0,atol=1e-14)
    assert np.isfinite(prediction).all()
    return {'experiment_id':name},result,metadata,prediction,sources
