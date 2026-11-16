"""State observability on complete causal prefixes of original heldout series."""
from pathlib import Path
import sys
import argparse
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next8
from src.next8.common import BASE,read_json,atomic,digest,utc_now,activity,note
from src.next8.cache import FeatureCache,immutable_npz
from src.next8.data import outer_scope
from src.next8.nested_auxiliary import early_scope
from src.next8.features import DETAILED
from src.next8.state_observability import evaluate_state_feature,load_state_archives


def main(fold,early):
    release=read_json(BASE/'PREDICTION_FREEZE_NEXT8.json');assert release['status']=='FROZEN'
    lock=read_json(BASE/'STATE_OBSERVABILITY_IMPLEMENTATION_LOCK_NEXT8.json')
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    cache=FeatureCache();scope=early_scope(outer_scope(fold,cache.data),early)
    directory=BASE/'state_observability'/scope.name;directory.mkdir(parents=True,exist_ok=True)
    if (directory/'DONE.json').exists():raise RuntimeError('Completed state observability cannot be rerun')
    with activity('Load frozen state channels '+scope.name,'STATE_PREDICTION_READ','B2_STATE_OBSERVABILITY'):
        state_archives=load_state_archives(cache,scope,release)
    files={}
    for j in DETAILED:
        p=directory/f'F{j:03d}';p.mkdir(parents=True,exist_ok=True)
        if (p/'DONE.json').exists():
            done=read_json(p/'DONE.json');assert digest(p/'GROUP_GAINS.npz')==done['group_gain_sha256']
        else:
            if any(p.iterdir()):raise RuntimeError('Partial observability output preserved')
            with activity(f'Current-state observability {scope.name} F{j:03d}','STATE_OBSERVABILITY','B2_STATE_OBSERVABILITY'):
                result,arrays=evaluate_state_feature(cache,scope,j,release,early=early,state_archives=state_archives)
                immutable_npz(p/'GROUP_GAINS.npz',**arrays)
                atomic(p/'DONE.json',{'timestamp':utc_now(),**result,'group_gain_sha256':digest(p/'GROUP_GAINS.npz')})
        files[p.name+'/DONE.json']=digest(p/'DONE.json')
    atomic(directory/'DONE.json',{'status':'COMPLETE','timestamp':utc_now(),'scope':scope.provenance(),'features':list(DETAILED),
        'files_sha256':files,'no_break_labels_used':True,'actual_M5_confident_condition_distinguished_from_probability_forecast':True})
    note('Complete current-state observability',scope.name+'; coverage, proper loss and abstentions all retained','B2_STATE_OBSERVABILITY')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=range(5),required=True);p.add_argument('--early',action='store_true')
    a=p.parse_args();main(a.fold,a.early)
