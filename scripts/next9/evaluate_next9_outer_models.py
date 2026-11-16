"""Once-frozen whole-group exact TS-AUC evaluation; no model fitting here."""
from pathlib import Path
import sys
import argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now
from src.next9.boundary import install_workspace_boundary


def main(fold):
    violations=install_workspace_boundary()
    lock=read_json(BASE/'OUTER_EVALUATION_IMPLEMENTATION_LOCK_NEXT9.json')
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    global_freeze=read_json(BASE/'GLOBAL_PREDICTIONS_FROZEN_NEXT9.json');assert global_freeze['status']=='FROZEN'
    release=read_json(BASE/'OUTER_RELEASE_NEXT9.json')
    assert release['global_prediction_freeze_sha256']==digest(BASE/'GLOBAL_PREDICTIONS_FROZEN_NEXT9.json')
    import numpy as np
    from src.next9.data import FeatureBank,outer_scope
    from src.next9.labels import ScopedLabels
    from src.next9.domains import masks
    from src.next9.inference import weights_for,rank_samples
    from src.next9.io import save_npz
    bank=FeatureBank();scope=outer_scope(bank,fold);labels=ScopedLabels(bank)
    root=BASE/'outer_model_evaluation'/scope.name
    if (root/'DONE.json').exists():return
    modes=[]
    for mode in ['NORMAL','EARLY']:
        member=next(r for r in global_freeze['records'] if r['fold']==fold and r['mode']==mode)
        path=BASE/member['path'];assert digest(path)==member['sha256'];frozen=read_json(path)
        directory=path.parent;assert digest(directory/'PREDICTIONS.npz')==frozen['prediction_sha256']
        with np.load(directory/'PREDICTIONS.npz') as z:
            rows=z['global_row'];pred={k:z[k] for k in frozen['methods']}
        y=labels.get(rows,scope,'FROZEN_OUTER_EVALUATION',prediction_receipt=frozen)
        group=np.asarray([bank.guard.split.groups[int(s)] for s in bank.row_ids[rows]],dtype=np.int32)
        results=[]
        for family,name,keep in masks(bank,rows,frozen['domain_spec']):
            out=root/mode/name
            if (out/'DONE.json').exists():results.append(read_json(out/'DONE.json'));continue
            with activity(f'{scope.name} {mode} {name} paired whole-original-group M-model evaluation','OUTER_EVALUATION','B6_BL80_REDUNDANCY'):
                yy,gg,tt=y[keep],group[keep],bank.times[rows[keep]]
                support=dict(rows=len(yy),groups=len(np.unique(gg)),positive_groups=len(np.unique(gg[yy==1])),negative_groups=len(np.unique(gg[yy==0])))
                major=min(support['positive_groups'],support['negative_groups'])>=64
                if not len(yy) or not len(np.unique(gg)) or min(support['positive_groups'],support['negative_groups'])==0:
                    point={k:dict(TS_AUC=None) for k in pred};samples={k:np.full(2000,np.nan) for k in pred}
                    resampling=dict(status='INSUFFICIENT_SUPPORT',draws=2000,reason='No observations or no opposite class groups')
                else:
                    unique=np.unique(gg);weights,resampling=weights_for(unique,f'OUTER_MODEL|{scope.name}|{mode}|{name}')
                    point={};samples={}
                    for method,p in pred.items():
                        a,b=rank_samples(yy,p[keep],tt,gg,unique,weights)
                        point[method]=dict(TS_AUC=None if not np.isfinite(a) else float(a));samples[method]=b
                    valid=np.logical_and.reduce([np.isfinite(v) for v in samples.values()])
                    resampling.update(status='SUPPORTED' if valid.sum()>=1900 else 'INSUFFICIENT_SUPPORT',valid_draws=int(valid.sum()),zero_pair_draws_not_imputed=True)
                save_npz(out/'BOOTSTRAP.npz',**samples)
                result=dict(status='COMPLETE',timestamp=utc_now(),fold=fold,mode=mode,scope=scope.provenance(),family=family,domain=name,
                    support=support,major_domain_supported=major,points=point,resampling=resampling,
                    bootstrap_sha256=digest(out/'BOOTSTRAP.npz'),prediction_freeze_sha256=digest(path),
                    global_freeze_sha256=digest(BASE/'GLOBAL_PREDICTIONS_FROZEN_NEXT9.json'),no_fitting_or_model_selection=True)
                atomic(out/'DONE.json',result);results.append(result)
        atomic(root/mode/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),scope=scope.provenance(),mode=mode,
            domains=[r['domain'] for r in results],domain_receipts_sha256={r['domain']:digest(root/mode/r['domain']/'DONE.json') for r in results}))
        modes.append(dict(mode=mode,sha256=digest(root/mode/'DONE.json')))
    assert not violations
    atomic(root/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),scope=scope.provenance(),modes=modes,
        label_access=labels.accesses,global_freeze_sha256=digest(BASE/'GLOBAL_PREDICTIONS_FROZEN_NEXT9.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',required=True,type=int,choices=range(5));main(p.parse_args().fold)
