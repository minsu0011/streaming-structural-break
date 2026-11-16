"""All fixed twelve features and three horizons, including failed controls."""
from pathlib import Path
import sys
import argparse
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next8
from src.next8.common import BASE,atomic,read_json,digest,utc_now,activity,note
from src.next8.cache import FeatureCache,immutable_npz
from src.next8.data import outer_scope
from src.next8.nested_auxiliary import early_scope
from src.next8.features import DETAILED,HORIZONS
from src.next8.forecast_evaluation import evaluate_forecast


def main(fold,early):
    release=read_json(BASE/'PREDICTION_FREEZE_NEXT8.json');assert release['status']=='FROZEN'
    lock=read_json(BASE/'FORECAST_SCORING_IMPLEMENTATION_LOCK_NEXT8.json')
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    cache=FeatureCache();scope=early_scope(outer_scope(fold,cache.data),early)
    directory=BASE/'forecast_evaluation'/scope.name;directory.mkdir(parents=True,exist_ok=True)
    if (directory/'DONE.json').exists():raise RuntimeError('Completed movement evaluation cannot be rerun')
    files={}
    for j in DETAILED:
        for h in HORIZONS:
            p=directory/f'F{j:03d}_h{h}';p.mkdir(parents=True,exist_ok=True)
            if (p/'DONE.json').exists():
                done=read_json(p/'DONE.json');assert done['scope']==scope.name
                assert digest(p/'GROUP_GAINS.npz')==done['group_gain_sha256']
            else:
                if any(p.iterdir()):raise RuntimeError('Partial forecast evaluation preserved')
                with activity(f'Evaluate movement and states {scope.name} F{j:03d} h{h}','MOVEMENT_EVALUATION','B3_SELF_HISTORY_SUFFICIENCY'):
                    result,arrays=evaluate_forecast(cache,scope,j,h,release)
                    immutable_npz(p/'GROUP_GAINS.npz',**arrays)
                    atomic(p/'DONE.json',{'timestamp':utc_now(),**result,'group_gain_sha256':digest(p/'GROUP_GAINS.npz')})
            files[p.name+'/DONE.json']=digest(p/'DONE.json')
    atomic(directory/'DONE.json',{'status':'COMPLETE','timestamp':utc_now(),'scope':scope.provenance(),
        'feature_horizon_pairs':len(DETAILED)*len(HORIZONS),'files_sha256':files,
        'all_horizons_and_rejected_methods_retained':True,'break_labels_used':False})
    note('Complete heldout movement/transition evaluation',scope.name+'; A remains separate from direction and model gain','B3_SELF_HISTORY_SUFFICIENCY')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=range(5),required=True);p.add_argument('--early',action='store_true')
    a=p.parse_args();main(a.fold,a.early)
