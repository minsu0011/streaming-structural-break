"""Read only original outer TRAIN labels for the fixed M6 eligibility gate."""
from pathlib import Path
import sys
import argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next8
import numpy as np
from src.next8.common import BASE,read_json,atomic,digest,utc_now,activity,note
from src.next8.cache import FeatureCache,immutable_npz
from src.next8.data import outer_scope,inner_scope,baseline_oof
from src.next8.integration import fixed_blend,integration_eligible
from src.next8.paired_bootstrap import evaluate_partition,summarize_paired


def main(fold):
    assert read_json(BASE/'FINAL_MODEL_TEST_GATE_NEXT8.json')['status']=='PASS'
    lock_path=BASE/'FINAL_MODEL_IMPLEMENTATION_LOCK_NEXT8.json';lock=read_json(lock_path)
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    cache=FeatureCache();outer=outer_scope(fold,cache.data)
    directory=BASE/'integration'/outer.name;directory.mkdir(parents=True,exist_ok=True)
    if (directory/'DONE.json').exists():raise RuntimeError('Do not re-open completed M6 eligibility')
    baseline=baseline_oof(fold,inner_training=True,data=cache.data)
    b_rows=baseline['global_row'];assert np.all(np.diff(b_rows)>0)
    all_points=[];all_samples=[];all_receipts=[];sources={}
    for inner in range(3):
        scope=inner_scope(fold,inner,cache.data);p=BASE/'models'/scope.name
        done=read_json(p/'DONE.json');assert done['scope']==scope.provenance()
        assert digest(p/'PREDICTIONS.npz')==done['files_sha256']['PREDICTIONS.npz']
        output=directory/f'inner_{inner}'
        if (output/'DONE.json').exists():
            saved=read_json(output/'DONE.json')
            assert saved['prediction_sha256']==digest(p/'PREDICTIONS.npz')
            assert digest(output/'BOOTSTRAP.npz')==saved['bootstrap_sha256']
            points=saved['points'];receipt=saved['resampling']
            with np.load(output/'BOOTSTRAP.npz',allow_pickle=False) as a:samples={k:a[k] for k in a.files}
        else:
            if output.exists() and any(output.iterdir()):raise RuntimeError('Partial inner uncertainty output preserved')
            output.mkdir(parents=True,exist_ok=True)
            with activity(f'Exact M6 inner group bootstrap outer{fold} inner{inner}','INNER_INTEGRATION_GATE','B6_BL80_REDUNDANCY'):
                with np.load(p/'PREDICTIONS.npz',allow_pickle=False) as a:
                    rows=a['global_row'];ids=a['dataset_id'];prediction=a['M5'].copy()
                scope.admit_validation(np.unique(ids));outer.admit_training(np.unique(ids))
                assert set(map(int,np.unique(ids)))==scope.validation_ids
                assert np.array_equal(cache.meta['dataset_id'][rows],ids)
                loc=np.searchsorted(b_rows,rows);assert np.array_equal(b_rows[loc],rows)
                reference=baseline['prediction'][loc]
                predictions={'M0':reference,'M5':prediction,'M6':fixed_blend(reference,prediction)}
                groups=np.asarray([cache.data.guard.split.groups[int(s)] for s in ids])
                points,samples,receipt=evaluate_partition(cache.meta['target'][rows],predictions,
                    cache.meta['time_online'][rows],groups,partition=f'outer{fold}_inner{inner}_M6',draws=2000)
                immutable_npz(output/'BOOTSTRAP.npz',**samples)
                atomic(output/'DONE.json',{'status':'COMPLETE','timestamp':utc_now(),'scope':scope.provenance(),
                    'points':points,'resampling':receipt,'prediction_sha256':digest(p/'PREDICTIONS.npz'),
                    'bootstrap_sha256':digest(output/'BOOTSTRAP.npz'),'all_labels_belong_to_outer_TRAIN':True})
        all_points.append(points);all_samples.append(samples);all_receipts.append(receipt)
        sources[p.relative_to(BASE).as_posix()+'/PREDICTIONS.npz']=digest(p/'PREDICTIONS.npz')
    comparison=summarize_paired(all_points,all_samples,'M0','M6')
    direct=summarize_paired(all_points,all_samples,'M0','M5')
    eligible=integration_eligible(comparison)
    atomic(directory/'DONE.json',{'status':'COMPLETE','timestamp':utc_now(),'scope':outer.provenance(),
        'eligible':eligible,'M6_status':'ELIGIBLE_FIXED_95_5' if eligible else 'NOT_RUN_NO_INDEPENDENT_INFO',
        'fixed_weights':{'BL80':.95,'M5':.05},'comparison':comparison,'M5_direct_comparison':direct,
        'points':all_points,'resampling':all_receipts,'source_prediction_sha256':sources,
        'eligibility_rule':'Mean three inner TS-AUC delta>=.0001,>=2/3 positive, exact paired group bootstrap CI lower>0',
        'outer_validation_labels_predictions_scores_used':False,'historically_adaptive_RP_not_independent_hidden_evidence':True})
    note('Complete fixed integration inner eligibility',outer.name+'; outer validation remains unopened','B6_BL80_REDUNDANCY')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=range(5),required=True)
    a=p.parse_args();main(a.fold)
