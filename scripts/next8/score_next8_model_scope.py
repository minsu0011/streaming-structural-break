"""Exact paired whole-group evaluation after the global prediction freeze."""
from pathlib import Path
import sys
import argparse
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next8
import numpy as np
from src.next8.common import BASE,atomic,read_json,digest,utc_now,activity,note
from src.next8.cache import FeatureCache,immutable_npz
from src.next8.data import outer_scope
from src.next8.nested_auxiliary import early_scope
from src.next8.evaluation import frozen_predictions,domains,exact_unique_partition


def main(fold,early):
    release=read_json(BASE/'PREDICTION_FREEZE_NEXT8.json');assert release['status']=='FROZEN'
    lock=read_json(BASE/'MODEL_SCORING_IMPLEMENTATION_LOCK_NEXT8.json')
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    cache=FeatureCache();scope=early_scope(outer_scope(fold,cache.data),early)
    rows,predictions,provenance=frozen_predictions(cache,scope,fold,early=early)
    source=f'models/{scope.name}/PREDICTIONS.npz'
    assert digest(BASE/source)==release['files_sha256'][source]
    directory=BASE/'model_evaluation'/scope.name;directory.mkdir(parents=True,exist_ok=True)
    if (directory/'DONE.json').exists():raise RuntimeError('Completed model scores cannot be silently rerun')
    ids=cache.meta['dataset_id'][rows];groups=np.asarray([cache.data.guard.split.groups[int(s)] for s in ids])
    y=cache.meta['target'][rows];t=cache.meta['time_online'][rows];records=[]
    maps=read_json(BASE/'EVALUATION_DOMAIN_MAPS_NEXT8.json')
    assert digest(BASE/'EVALUATION_DOMAIN_MAPS_NEXT8.json')==release['files_sha256']['EVALUATION_DOMAIN_MAPS_NEXT8.json']
    for family,name,keep,definition in domains(cache,scope,rows,maps['scopes'][scope.name]):
        p=directory/name;p.mkdir(parents=True,exist_ok=True)
        if (p/'DONE.json').exists():
            record=read_json(p/'DONE.json');assert record['scope']==scope.provenance()
            assert digest(p/'BOOTSTRAP.npz')==record['bootstrap_sha256'];records.append(record);continue
        if any(p.iterdir()):raise RuntimeError('Partial exact bootstrap output preserved for explicit repair')
        with activity(f'Exact outer score {scope.name} {name}','HELDOUT_MODEL_EVALUATION','B6_BL80_REDUNDANCY'):
            point,sample,receipt=exact_unique_partition(y[keep],{m:a[keep] for m,a in predictions.items()},t[keep],groups[keep],
                partition=scope.name+'_'+name,draws=2000,require_support=family!='ALL')
            immutable_npz(p/'BOOTSTRAP.npz',**sample)
            record={'status':'COMPLETE','timestamp':utc_now(),'scope':scope.provenance(),'family':family,'domain':name,
                'definition':definition,'points':point,'resampling':receipt,'bootstrap_sha256':digest(p/'BOOTSTRAP.npz'),
                'prediction_provenance':provenance,'no_fitting_or_selection_after_outer_score_release':True}
            atomic(p/'DONE.json',record);records.append(record)
    atomic(directory/'DONE.json',{'status':'COMPLETE','timestamp':utc_now(),'scope':scope.provenance(),
        'prediction_provenance':provenance,'domains':[r['domain'] for r in records],
        'files_sha256':{r['domain']+'/DONE.json':digest(directory/r['domain']/'DONE.json') for r in records},
        'prediction_freeze_sha256':digest(BASE/'PREDICTION_FREEZE_NEXT8.json'),'scoring_only_no_fit':True})
    note('Complete exact model/domain evaluation',scope.name+'; retain all methods and sparse domains','B6_BL80_REDUNDANCY')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=range(5),required=True);p.add_argument('--early',action='store_true')
    a=p.parse_args();main(a.fold,a.early)
