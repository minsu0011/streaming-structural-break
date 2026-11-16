"""Honest fixed diagnostic, followed by one offset fit or exact BL80 fallback."""
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
    lock=read_json(BASE/'META_CORRECTION_IMPLEMENTATION_LOCK_NEXT9.json')
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    import numpy as np
    import joblib
    from src.next9.data import FeatureBank,outer_scope,inner_scope
    from src.next9.labels import ScopedLabels
    from src.next9.baseline import baseline_for
    from src.next9.meta_correction import validate_oof,fixed_diagnostic,fit_correction,eligible
    from src.next9.inference import weights_for,rank_samples
    from src.next8.paired_bootstrap import summarize_paired
    from src.next9.io import save_npz
    bank=FeatureBank();outer=outer_scope(bank,fold);labels=ScopedLabels(bank)
    root=BASE/'meta_correction'/outer.name
    if (root/'DONE.json').exists():return
    mode_receipts=[]
    for mode in ['NORMAL','EARLY']:
        out=root/mode
        if (out/'DONE.json').exists():mode_receipts.append(dict(mode=mode,sha256=digest(out/'DONE.json')));continue
        with activity(f'{outer.name} {mode} complete-procedure inner qualification for fixed-offset M7','META_QUALIFICATION','B6_BL80_REDUNDANCY'):
            parts=[];constant=[];sources=[]
            expected=bank.sampled_rows(outer.training_ids,cutoff=199 if mode=='EARLY' else None,stride=1)
            for k in range(3):
                scope=inner_scope(bank,fold,k);directory=BASE/'joint_heads'/scope.name/mode
                done=read_json(directory/'DONE.json');spec=read_json(directory/'SPEC_FROZEN.json')
                assert done['scope']==scope.provenance()==spec['scope']
                for name,sha in done['files_sha256'].items():assert digest(directory/name)==sha
                alias=done['aliases']['M5'];model=joblib.load(directory/(alias+'.joblib'))
                selection=BASE/'orientation_selection'/scope.name/mode/'SELECTION_FROZEN.json'
                selected=read_json(selection)
                assert selected['scope']==scope.provenance()
                assert model.training['selection_sha256']==digest(selection)
                assert read_json(BASE/'candidates'/scope.name/'FROZEN.json')['scope']==scope.provenance()
                assert read_json(BASE/'self_transitions'/scope.name/mode/'DONE.json')['scope']==scope.provenance()
                if mode=='EARLY':assert model.training['early_training_maximum']==199
                with np.load(directory/'PREDICTIONS.npz') as z:
                    rows=z['global_row'];p=z['M5'];ids=z['dataset_id']
                keep=(bank.times[rows]<=199) if mode=='EARLY' else np.ones(len(rows),dtype=bool)
                rows,p,ids=rows[keep],p[keep],ids[keep]
                np.testing.assert_array_equal(ids,bank.row_ids[rows])
                parts.append(dict(scope=scope,dataset_id=ids,global_row=rows,prediction=p,
                    training_provenance=model.training,complete_selection_excluded=True))
                constant.append(model.estimator is None and float(np.ptp(p))==0.)
                sources.append(dict(scope=scope.provenance(),head_sha256=digest(directory/(alias+'.joblib')),
                    prediction_sha256=done['files_sha256']['PREDICTIONS.npz'],selection_sha256=digest(selection)))
            meta=validate_oof(outer,parts,expected)
            if all(constant):
                qualification=dict(status='SUPPORTED',mean_delta=0.,fold_deltas=[0.,0.,0.],positive_folds=0,ci_low=0.,ci_high=0.,
                    reason='Exact within-partition rank invariance for .95BL+.05constant; no outcome-dependent fitting or baseline access needed')
                correction=None;baseline=None;quality=[]
            else:
                if mode=='NORMAL':baseline,baseline_source=baseline_for(bank,outer,fold,expected,inner_oof=True)
                else:
                    rr=[];pp=[];baseline_source=[]
                    for k in range(3):
                        d=BASE/'early_inner_reference'/f'outer_{fold}_inner_{k}';r=read_json(d/'DONE.json')
                        assert r['required'] and r['reference_fits']==2
                        assert digest(d/'BL80_INNER_EARLY.npz')==r['prediction_sha256']
                        with np.load(d/'BL80_INNER_EARLY.npz') as z:rr.append(z['global_row']);pp.append(z['prediction'])
                        baseline_source.append(dict(path=str(d.relative_to(BASE)),sha256=r['prediction_sha256']))
                    rr,pp=np.concatenate(rr),np.concatenate(pp);order=np.argsort(rr)
                    np.testing.assert_array_equal(rr[order],expected);baseline=pp[order]
                points=[];samples=[];quality=[]
                for k,part in enumerate(parts):
                    rows=part['global_row'];loc=np.searchsorted(expected,rows)
                    np.testing.assert_array_equal(expected[loc],rows)
                    scope=part['scope'];y=labels.get(rows,scope,'INTERNAL_VALIDATION',early=mode=='EARLY')
                    group=np.asarray([bank.guard.split.groups[int(s)] for s in bank.row_ids[rows]],dtype=np.int32)
                    unique=np.unique(group);weights,receipt=weights_for(unique,f'M7_QUALIFY|{outer.name}|{mode}|inner{k}')
                    point={};boot={}
                    for name,pred in [('M0',baseline[loc]),('FIXED95',fixed_diagnostic(baseline[loc],part['prediction']))]:
                        a,b=rank_samples(y,pred,bank.times[rows],group,unique,weights)
                        point[name]=dict(TS_AUC=None if not np.isfinite(a) else float(a));boot[name]=b
                    points.append(point);samples.append(boot);quality.append(dict(inner=k,points=point,bootstrap=receipt))
                qualification=summarize_paired(points,samples,'M0','FIXED95')
                atomic(out/'FIXED_DIAGNOSTIC_EVIDENCE.json',dict(status='COMPLETE',qualification=qualification,panels=quality,
                    baseline_sources=baseline_source,learned_M7_not_fitted_or_evaluated_yet=True))
                if eligible(qualification):
                    y=labels.get(expected,outer,'TRAIN',early=mode=='EARLY')
                    group=np.asarray([bank.guard.split.groups[int(s)] for s in bank.row_ids[expected]],dtype=np.int32)
                    correction=fit_correction(baseline,meta,y,bank.row_ids[expected],group,outer,qualification)
                    joblib.dump(correction,out/'CORRECTION.joblib')
                else:correction=None
            outer_dir=BASE/'joint_heads'/outer.name/mode
            done=read_json(outer_dir/'DONE.json');assert digest(outer_dir/'PREDICTIONS.npz')==done['files_sha256']['PREDICTIONS.npz']
            with np.load(outer_dir/'PREDICTIONS.npz') as z:rows=z['global_row'];head=z['M5']
            keep=(bank.times[rows]>=400) if mode=='EARLY' else np.ones(len(rows),dtype=bool)
            rows,head=rows[keep],head[keep]
            outer_bl,outer_source=baseline_for(bank,outer,fold,rows,early=mode=='EARLY')
            prediction=outer_bl.copy() if correction is None else correction.predict(outer_bl,head)
            save_npz(out/'PREDICTIONS.npz',global_row=rows,dataset_id=bank.row_ids[rows],M0=outer_bl,M7=prediction)
            atomic(out/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),scope=outer.provenance(),mode=mode,
                qualification=qualification,eligible=correction is not None,M7_alias='M0' if correction is None else None,
                correction_sha256=None if correction is None else digest(out/'CORRECTION.joblib'),
                correction_training=None if correction is None else correction.training,
                prediction_sha256=digest(out/'PREDICTIONS.npz'),outer_baseline_sources=outer_source,inner_sources=sources,
                outer_labels_used=False,outer_scores_released=False,learned_meta_resubstitution_score_never_used=True))
            mode_receipts.append(dict(mode=mode,sha256=digest(out/'DONE.json')))
    assert not violations
    atomic(root/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),scope=outer.provenance(),records=mode_receipts,
        label_access=labels.accesses,outer_scores_released=False,implementation_sha256=digest(BASE/'META_CORRECTION_IMPLEMENTATION_LOCK_NEXT9.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',required=True,type=int,choices=range(5));main(p.parse_args().fold)
