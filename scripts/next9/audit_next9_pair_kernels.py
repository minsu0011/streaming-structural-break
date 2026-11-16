"""Independently reconstruct every reported outer model domain and 2000 draws."""
from pathlib import Path
import sys,argparse,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now
from src.next9.boundary import install_workspace_boundary


def main(fold,mode):
    denied=install_workspace_boundary()
    import numpy as np
    from src.next9.data import FeatureBank,outer_scope
    from src.next9.labels import ScopedLabels
    from src.next9.domains import masks
    from src.next9.inference import weights_for
    from src.next9.pair_kernel_audit import prepare_pairs,weighted_pairs
    bank=FeatureBank();scope=outer_scope(bank,fold);labels=ScopedLabels(bank)
    global_path=BASE/'GLOBAL_PREDICTIONS_FROZEN_NEXT9.json';global_freeze=read_json(global_path)
    assert global_freeze['status']=='FROZEN'
    member=next(r for r in global_freeze['records'] if r['fold']==fold and r['mode']==mode)
    frozen_path=BASE/member['path'];assert digest(frozen_path)==member['sha256']
    frozen=read_json(frozen_path);source=frozen_path.parent/'PREDICTIONS.npz'
    assert digest(source)==frozen['prediction_sha256']
    with np.load(source) as z:
        rows=z['global_row'];pred={k:z[k] for k in frozen['methods']}
    y=labels.get(rows,scope,'FROZEN_OUTER_EVALUATION',prediction_receipt=frozen)
    groups=np.asarray([bank.guard.split.groups[int(s)] for s in bank.row_ids[rows]],dtype=np.int32)
    root=BASE/'pair_kernel_audit'/scope.name/mode
    if (root/'DONE.json').exists():raise FileExistsError('Already audited; do not credit rerun')
    records=[]
    for family,name,keep in masks(bank,rows,frozen['domain_spec']):
        target=root/(name+'.json')
        if target.exists():records.append(read_json(target));continue
        with activity(f'{scope.name} {mode} {name} explicit pair kernels verify all 2000 bootstrap draws',
                      'INDEPENDENT_STATISTIC_AUDIT','B9_NORMALIZATION_STATE_AVAILABILITY'):
            old_root=BASE/'outer_model_evaluation'/scope.name/mode/name
            old=read_json(old_root/'DONE.json');assert old['status']=='COMPLETE'
            assert old['prediction_freeze_sha256']==digest(frozen_path)
            assert digest(old_root/'BOOTSTRAP.npz')==old['bootstrap_sha256']
            yy,gg,tt=y[keep],groups[keep],bank.times[rows[keep]]
            unique=np.unique(gg);resampling={};method_records=[];computed={}
            if len(unique):
                weights,resampling=weights_for(unique,f'OUTER_MODEL|{scope.name}|{mode}|{name}')
                for key in ['weight_sha256','resample_key_sha256']:
                    if key in old['resampling']:assert resampling[key]==old['resampling'][key]
                weights=np.vstack([np.ones((1,len(unique)),dtype=np.uint16),weights])
            with np.load(old_root/'BOOTSTRAP.npz') as previous:
                for method,values in pred.items():
                    p=values[keep];key=hashlib.sha256(p.tobytes()).hexdigest()
                    if key in computed:
                        old_p,actual,pairs= computed[key]
                        assert np.array_equal(p,old_p)
                        alias=True
                    else:
                        ug,count,win2=prepare_pairs(yy,p,tt,gg)
                        np.testing.assert_array_equal(ug,unique)
                        if len(unique):actual,pairs=weighted_pairs(count,win2,weights)
                        else:actual=np.full(2001,np.nan);pairs=np.zeros(2001)
                        computed[key]=(p,actual,pairs);alias=False
                    expected=old['points'][method]['TS_AUC']
                    if expected is None:assert np.isnan(actual[0])
                    else:assert np.float64(actual[0]).tobytes()==np.float64(expected).tobytes(),(name,method,'point')
                    old_samples=previous[method]
                    assert np.array_equal(np.isnan(actual[1:]),np.isnan(old_samples))
                    finite=np.isfinite(old_samples)
                    assert actual[1:][finite].tobytes()==old_samples[finite].tobytes(),(name,method,'bootstrap')
                    method_records.append(dict(method=method,point=expected,point_bitwise=True,all_2000_draws_bitwise=True,
                        finite_draws=int(finite.sum()),same_time_pairs=int(pairs[0]),exact_prediction_alias_reused=alias,
                        reconstructed_bootstrap_sha256=hashlib.sha256(actual[1:].tobytes()).hexdigest()))
            record=dict(status='PASS',timestamp=utc_now(),fold=fold,mode=mode,domain=name,family=family,
                rows=len(yy),groups=len(unique),methods=method_records,unique_prediction_vectors=len(computed),
                original_receipt_sha256=digest(old_root/'DONE.json'),original_bootstrap_sha256=old['bootstrap_sha256'],
                resampling=resampling,algorithm='Explicit positive-negative same-time pair integer matrices and group-weight quadratic forms; no score ranks.',
                count_arithmetic_exact_below_2pow53=True,zero_pair_draws_remain_nan=True)
            atomic(target,record);records.append(record)
    assert not denied
    atomic(root/'DONE.json',dict(status='PASS',timestamp=utc_now(),fold=fold,mode=mode,scope=scope.provenance(),
        domains=len(records),domain_receipts_sha256={r['domain']:digest(root/(r['domain']+'.json')) for r in records},
        label_access=labels.accesses,no_fitting_or_selection=True,source_prediction_freeze_sha256=digest(frozen_path)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,required=True,choices=range(5));p.add_argument('--mode',required=True,choices=['NORMAL','EARLY'])
    a=p.parse_args();main(a.fold,a.mode)
