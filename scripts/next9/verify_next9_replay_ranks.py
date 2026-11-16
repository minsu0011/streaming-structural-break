"""Independent rankdata-based verification of live versus frozen TS-AUC."""
from pathlib import Path
import sys
import argparse
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now
from src.next9.boundary import install_workspace_boundary


def main(fold):
    violations=install_workspace_boundary()
    lock=read_json(BASE/'REPLAY_RANK_VERIFICATION_LOCK_NEXT9.json')
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    import numpy as np
    from src.next9.data import FeatureBank,outer_scope
    from src.next9.labels import ScopedLabels
    from src.next9.domains import masks
    from src.scoring.ts_auc import ts_auc
    bank=FeatureBank();scope=outer_scope(bank,fold);labels=ScopedLabels(bank)
    root=BASE/'replay_rank_verification'/scope.name
    if (root/'DONE.json').exists():return
    global_path=BASE/'GLOBAL_PREDICTIONS_FROZEN_NEXT9.json';global_freeze=read_json(global_path);modes=[]
    assert global_freeze['status']=='FROZEN'
    for mode in ['NORMAL','EARLY']:
        member=next(r for r in global_freeze['records'] if r['fold']==fold and r['mode']==mode)
        path=BASE/member['path'];assert digest(path)==member['sha256'];frozen=read_json(path)
        assert digest(path.parent/'PREDICTIONS.npz')==frozen['prediction_sha256']
        with np.load(path.parent/'PREDICTIONS.npz') as z:rows=z['global_row'];expected={k:z[k] for k in frozen['methods']}
        actual={k:[] for k in expected};actual_rows=[]
        for part in [0,1]:
            source=BASE/'live_replay'/scope.name/mode/f'part_{part}';done=read_json(source/'DONE.json')
            assert digest(source/'PREDICTIONS.npz')==done['prediction_sha256']
            with np.load(source/'PREDICTIONS.npz') as z:
                actual_rows.append(z['global_row'])
                for k in actual:actual[k].append(z[k])
        actual_rows=np.concatenate(actual_rows);order=np.argsort(actual_rows);np.testing.assert_array_equal(actual_rows[order],rows)
        actual={k:np.concatenate(v)[order] for k,v in actual.items()};y=labels.get(rows,scope,'FROZEN_OUTER_EVALUATION',prediction_receipt=frozen)
        records=[]
        for family,name,keep in masks(bank,rows,frozen['domain_spec']):
            with activity(f'{scope.name} {mode} {name} independent rankdata live/frozen TS-AUC parity','RANK_REPRODUCTION','B9_NORMALIZATION_STATE_AVAILABILITY'):
                original=read_json(BASE/'outer_model_evaluation'/scope.name/mode/name/'DONE.json')
                for method in expected:
                    score,details=ts_auc(y[keep],actual[method][keep],bank.times[rows[keep]],return_details=True)
                    pairs=sum(r['weight'] for r in details)
                    ref,refdetails=ts_auc(y[keep],expected[method][keep],bank.times[rows[keep]],return_details=True)
                    assert sum(r['weight'] for r in refdetails)==pairs
                    authoritative=original['points'][method]['TS_AUC']
                    if pairs:
                        assert authoritative is not None and abs(ref-authoritative)<=1e-12,(fold,mode,name,method,'independent scorer')
                        error=abs(score-ref);assert error<=1e-8,(fold,mode,name,method,error)
                    else:
                        assert authoritative is None
                        score=ref=None;error=None
                    records.append(dict(fold=fold,mode=mode,family=family,domain=name,method=method,opposite_class_pairs=pairs,
                        live_TS_AUC=score,frozen_TS_AUC=ref,TS_AUC_absolute_error=error,
                        status='PASS' if pairs else 'UNDEFINED_NO_PAIRS',no_pair_score_not_imputed=True))
        atomic(root/mode/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),scope=scope.provenance(),mode=mode,records=records,
            maximum_TS_AUC_error=max((r['TS_AUC_absolute_error'] for r in records if r['TS_AUC_absolute_error'] is not None),default=0.),
            probability_tolerance_and_rank_tolerance_checked_separately=True))
        modes.append(dict(mode=mode,sha256=digest(root/mode/'DONE.json')))
    assert not violations
    atomic(root/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),scope=scope.provenance(),modes=modes,
        label_access=labels.accesses,no_refitting_or_selection=True,global_freeze_sha256=digest(global_path)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',required=True,type=int,choices=range(5));main(p.parse_args().fold)
