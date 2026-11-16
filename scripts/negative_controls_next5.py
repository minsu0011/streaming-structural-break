"""One fixed train-only permutation plus explicit invalid-future falsification."""
from pathlib import Path
import argparse
import hashlib
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from numba import set_num_threads
from src.next5.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next5.data import prepare_research,admitted_npz
from src.next5.cache import build_design
from src.next5.model import fit_registered
from src.next5.bootstrap import PreparedAUC
from src.next5.scheduler import register_job
from src.next4.negative_controls import permute_training_labels_within_time,detect_invalid_future_controls
from src.scoring.ts_auc import ts_auc


class Online:
    def __init__(self,points):self.points=points;self.consumed=0;self.iterated=False
    def __iter__(self):
        if self.iterated:raise AssertionError('Online iterator traversed twice')
        self.iterated=True;return self
    def __next__(self):
        if self.consumed==len(self.points):raise StopIteration
        value=self.points[self.consumed];self.consumed+=1;return value
    def __len__(self):raise AssertionError('Future length requested')
    def __length_hint__(self):raise AssertionError('Future length hint requested')
    def __array__(self,*args,**kwargs):raise AssertionError('Future suffix materialized')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--register',action='store_true');args=parser.parse_args()
    cid='REF_NEGATIVE_LABEL_PERMUTATION';regpath=BASE/'references/registrations'/(cid+'.json');policy_path=BASE/'negative_controls/POLICY.json'
    if args.register:
        original=read_json(BASE/'references/registrations/REF_J102.json')
        registration={**original,'experiment_id':cid,'created_utc':utc_now(),'use':'One full ResearchPool fold-0 label-permutation negative control; never a candidate',
            'negative_control':True,'training_label_change':'Permute only training targets within online-time strata with seed 20260908'}
        atomic_json(regpath,registration,immutable=True)
        atomic_json(policy_path,{'created_utc':utc_now(),'status':'FROZEN_BEFORE_CONTROL_SCORE','registration_sha256':digest(regpath),
            'seed':read_json(ROOT/'NEXT5_RANDOMNESS_LOCK.json')['label_permutation_seed'],'scope':'Full ResearchPool PRIMARY fold0',
            'new_control_fits':1,'additional_permutations_permitted':False,'constant_score':.5,'selection_authority':False,
            'sources_sha256':{n:digest(ROOT/n) for n in ('scripts/negative_controls_next5.py','src/next4/negative_controls.py','src/next5/bootstrap.py')},
            'bootstrap_counts':'Existing frozen full PRIMARY fold0 counts, no new real-data draws'},immutable=True)
        register_job({'id':'NC001_LABEL_PERMUTATION','kind':'validation','priority':45,'depends_on_success':['REF_J102_full','DIAG002_BOOTSTRAP_CONTRACT'],
            'argv':['scripts/negative_controls_next5.py'],'description':'One fixed ResearchPool train-time-stratified label permutation, constant/time-only score and future traps','retry_safe':False})
        progress('Negative-control fit preregistered','Exactly one J102 label permutation within training time strata; never selection evidence');return
    policy=read_json(policy_path)
    for name,sha in policy['sources_sha256'].items():assert digest(ROOT/name)==sha
    assert digest(regpath)==policy['registration_sha256'];registration=read_json(regpath)
    out=BASE/'negative_controls/label_permutation_v1';out.mkdir(parents=True,exist_ok=False)
    data=prepare_research();reference=read_json(BASE/'references/registrations/REF_J102.json');x,metadata,cache=build_design(data,reference)
    train=np.flatnonzero(metadata['fold']!=0);valid=np.flatnonzero(metadata['fold']==0)
    altered=permute_training_labels_within_time(metadata,train,policy['seed'])
    np.testing.assert_array_equal(altered['target'][valid],metadata['target'][valid])
    times=metadata['time_online'];before=np.bincount(times[train],weights=metadata['target'][train]);after=np.bincount(times[train],weights=altered['target'][train])
    np.testing.assert_array_equal(before,after);changed=int(np.sum(metadata['target'][train]!=altered['target'][train]));assert changed>0
    atomic_json(out/'FIT_START.json',{'created_utc':utc_now(),'intended_new_fits':1,'training_rows':len(train),'validation_rows':len(valid),
        'registration_sha256':digest(regpath),'changed_targets':changed,'permuted_train_target_sha256':hashlib.sha256(altered['target'][train].tobytes()).hexdigest()},immutable=True)
    progress('Single label-permutation fit','Training-only targets changed; held-out targets and time-stratified class counts preserved')
    model,training=fit_registered(x,altered,train,valid,guard=data.guard,registration=registration)
    model.save(out/'model.joblib');prediction=model.predict(x[valid]);target=metadata['target'][valid];online_time=times[valid]
    score=float(ts_auc(target,prediction,online_time));constant=float(ts_auc(target,np.full(len(valid),.5),online_time))
    time_only=float(ts_auc(target,online_time/(1.+online_time),online_time));assert constant==time_only==.5
    invalid=detect_invalid_future_controls(Online,np.arange(10,dtype=np.float32));assert len(invalid)==3 and all(invalid.values())
    lock=read_json(BASE/'bootstrap/COUNTS_LOCK.json');receipt=lock['folds'][0];path=ROOT/receipt['path'];assert digest(path)==receipt['sha256']
    draws=admitted_npz(path,data.guard,'Negative-control existing paired counts');ids=metadata['dataset_id'][valid]
    code=np.searchsorted(draws['dataset_id'],ids).astype(np.int32);np.testing.assert_array_equal(draws['dataset_id'][code],ids)
    set_num_threads(2);values=PreparedAUC.create(target,prediction,online_time,code,draws['counts'].shape[1]).evaluate(draws['counts'])
    assert abs(values[0]-score)<1e-14
    np.savez(out/'predictions.npz',dataset_id=ids,global_row=valid,prediction=prediction);np.save(out/'BOOTSTRAP_SCORES.npy',values,allow_pickle=False)
    atomic_json(out/'RESULTS.json',{'created_utc':utc_now(),'status':'PASS','new_fits':1,'training':training,
        'permuted_TS_AUC':score,'conditional_CI':[float(np.quantile(values[1:],.025)),float(np.quantile(values[1:],.975))],
        'constant_TS_AUC':constant,'time_only_TS_AUC':time_only,'future_controls_detected':invalid,
        'changed_train_targets':changed,'held_out_targets_unchanged':True,'train_time_class_counts_preserved':True,
        'shadow_rows':0,'seal_rows':0,'selection_authority':False,'additional_seed_search':False,
        'interpretation':'One falsification control. Report observed chance deviation; no calibrated global no-leakage claim from one fit.',
        'policy_sha256':digest(policy_path),'files_sha256':{p.name:digest(p) for p in out.iterdir() if p.is_file()}},immutable=True)
    data.guard.check_state();progress('Limited negative controls complete','One permutation fit, constant/time-only TS-AUC exactly 0.5, three deliberately invalid future operations detected')
    print('NEXT5 NEGATIVE CONTROLS',score,'constant',constant,'time_only',time_only,flush=True)


if __name__=='__main__':main()
