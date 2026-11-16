from pathlib import Path
import sys
import argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now
from src.next9.reference_access import install_reference_boundary


def main(fold,inner):
    violations=install_reference_boundary()
    lock=read_json(BASE/'EARLY_INNER_REFERENCE_IMPLEMENTATION_LOCK_NEXT9.json')
    for name,sha in lock['source_sha256'].items(): assert digest(ROOT/name)==sha,name
    import numpy as np
    import joblib
    from src.next9.data import FeatureBank,inner_scope
    from src.next9.labels import ScopedLabels
    from src.next9.early_reference import fit
    from src.next9.io import save_npz
    bank=FeatureBank();scope=inner_scope(bank,fold,inner)
    out=BASE/'early_inner_reference'/scope.name
    if (out/'DONE.json').exists():return
    head=read_json(BASE/'joint_heads'/scope.name/'EARLY'/'DONE.json')
    alias=head['aliases']['M5'];head_path=BASE/'joint_heads'/scope.name/'EARLY'/(alias+'.joblib')
    assert digest(head_path)==head['files_sha256'][alias+'.joblib']
    all_constant=True
    for k in range(3):
        other=BASE/'joint_heads'/f'outer_{fold}_inner_{k}'/'EARLY'
        receipt=read_json(other/'DONE.json');method=receipt['aliases']['M5']
        path=other/(method+'.joblib');assert digest(path)==receipt['files_sha256'][method+'.joblib']
        all_constant &= joblib.load(path).estimator is None
    if all_constant:
        atomic(out/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),scope=scope.provenance(),required=False,
            reason='All three inner M5 heads are constant; fixed .95BL+.05constant leaves every within-partition TS rank unchanged',
            fixed_diagnostic_exact_delta=0.,reference_fits=0,head_model_sha256=digest(head_path)))
        return
    labels=ScopedLabels(bank)
    rows=bank.sampled_rows(scope.validation_ids,cutoff=199,stride=1)
    pieces=[];records=[]
    for kind in ['J102','C205']:
        directory=out/kind
        with activity(f'{scope.name} early-only excluded {kind} original reference recipe','EARLY_REFERENCE','B6_BL80_REDUNDANCY'):
            fitted,source=fit(kind,bank,labels,scope)
            directory.mkdir(parents=True,exist_ok=True);fitted.save(directory/'model.joblib')
            pred=np.empty(len(rows))
            for start in range(0,len(rows),32768):pred[start:start+32768]=fitted.predict(source[rows[start:start+32768]])
            save_npz(directory/'PREDICTIONS.npz',global_row=rows,dataset_id=bank.row_ids[rows],prediction=pred)
            done=dict(status='COMPLETE',timestamp=utc_now(),kind=kind,scope=scope.provenance(),training=fitted.training,
                model_sha256=digest(directory/'model.joblib'),prediction_sha256=digest(directory/'PREDICTIONS.npz'))
            atomic(directory/'DONE.json',done);records.append(done);pieces.append(pred)
            del source,fitted
    save_npz(out/'BL80_INNER_EARLY.npz',global_row=rows,dataset_id=bank.row_ids[rows],prediction=.8*pieces[0]+.2*pieces[1])
    assert not violations
    atomic(out/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),scope=scope.provenance(),required=True,
        reference_fits=2,records=records,prediction_sha256=digest(out/'BL80_INNER_EARLY.npz'),
        label_access=labels.accesses,validation_scores_released=False,
        implementation_sha256=digest(BASE/'EARLY_INNER_REFERENCE_IMPLEMENTATION_LOCK_NEXT9.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',required=True,type=int,choices=range(5));p.add_argument('--inner',required=True,type=int,choices=range(3))
    a=p.parse_args();main(a.fold,a.inner)
