"""Entire153 bank atlas and fixed selected-feature direction comparisons."""
from pathlib import Path
import sys
import argparse
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next8
from src.next8.common import BASE,atomic,read_json,digest,utc_now,activity,note
from src.next8.cache import FeatureCache,immutable_npz
from src.next8.data import outer_scope
from src.next8.nested_auxiliary import early_scope
from src.next8.screening import sampled_rows
from src.next8.features import N_FEATURES
from src.next8.feature_evaluation import evaluate_feature


def main(fold,early):
    release=read_json(BASE/'PREDICTION_FREEZE_NEXT8.json');assert release['status']=='FROZEN'
    lock=read_json(BASE/'FEATURE_SCORING_IMPLEMENTATION_LOCK_NEXT8.json')
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    cache=FeatureCache();scope=early_scope(outer_scope(fold,cache.data),early)
    sp=BASE/'h_context_selection'/scope.name/'SELECTION.json';assert digest(sp)==release['files_sha256'][sp.relative_to(BASE).as_posix()]
    selection=read_json(sp)
    for name in ('DIRECTION_DOMAIN_LOCK_NEXT8.json','EVALUATION_DOMAIN_MAPS_NEXT8.json'):assert digest(BASE/name)==release['files_sha256'][name]
    definitions=read_json(BASE/'DIRECTION_DOMAIN_LOCK_NEXT8.json')['scopes'][scope.name]['records']
    specs=read_json(BASE/'EVALUATION_DOMAIN_MAPS_NEXT8.json')['scopes'][scope.name]
    rows=sampled_rows(cache,scope.validation_ids)
    if early:rows=rows[cache.meta['time_online'][rows]>=400]
    directory=BASE/'feature_evaluation'/scope.name;directory.mkdir(parents=True,exist_ok=True)
    if (directory/'DONE.json').exists():raise RuntimeError('Completed feature evaluation cannot be rerun')
    files={}
    for j in range(N_FEATURES):
        p=directory/f'F{j:03d}';p.mkdir(parents=True,exist_ok=True)
        if (p/'DONE.json').exists():
            done=read_json(p/'DONE.json');assert done['scope']==scope.provenance()
            assert digest(p/'GROUP_EVIDENCE.npz')==done['group_evidence_sha256']
        else:
            if any(p.iterdir()):raise RuntimeError('Partial feature evaluation preserved for explicit repair')
            with activity(f'Heldout feature directions {scope.name} F{j:03d}','FEATURE_EVALUATION','B1_CONDITIONAL_DIRECTION'):
                result,arrays=evaluate_feature(cache,scope,j,rows,selection,specs,definitions,release,early=early)
                immutable_npz(p/'GROUP_EVIDENCE.npz',**arrays)
                atomic(p/'DONE.json',{'timestamp':utc_now(),**result,'group_evidence_sha256':digest(p/'GROUP_EVIDENCE.npz')})
        files[p.name+'/DONE.json']=digest(p/'DONE.json')
    atomic(directory/'DONE.json',{'status':'COMPLETE','timestamp':utc_now(),'scope':scope.provenance(),
        'features':N_FEATURES,'files_sha256':files,'no_selection_feedback':True,
        'feature_bank_atlas_and_BL_conditioned_trials_all_retained':True,'certification_requires_cross_fold_aggregation':True})
    note('Complete all-bank heldout feature audit',scope.name+'; no single fold certification','B1_CONDITIONAL_DIRECTION')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=range(5),required=True);p.add_argument('--early',action='store_true')
    a=p.parse_args();main(a.fold,a.early)
