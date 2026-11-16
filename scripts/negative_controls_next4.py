"""One prospectively locked label-permutation fit and invalid-future controls."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys
import hashlib
import numpy as np
from numba import set_num_threads
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next4.negative_controls import permute_training_labels_within_time,detect_invalid_future_controls
from src.next4.model import fit_base
from src.next4.cache import build_design
from src.next4.frozen import fold_predictions
from src.next.data import prepare_dev
from src.next2.blend import combine
from src.scoring.ts_auc import ts_auc
from src.next.bootstrap import PreparedAUC,paired_summary
from src.validation.group_bootstrap import draw_group_multiplicities
from scripts.verify_next2_shared_public_api import Online


def main():
    set_num_threads(2);policy_path=BASE/'diagnostic_policy/NEGATIVE_CONTROL_POLICY.json';policy=read_json(policy_path)
    for name,sha in policy['sources_sha256'].items():assert digest(ROOT/name)==sha
    candidate='B2_LOG_ENERGY_SEMITAIL';rp=BASE/'registrations'/(candidate+'.json');reg=read_json(rp)
    assert digest(rp)==policy['registration_sha256']
    out=BASE/'negative_controls/label_permutation_v1';out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'START.json',{'created_utc':utc_now(),'policy_sha256':digest(policy_path),'intended_new_fits':1},immutable=True)
    data=prepare_dev(ROOT);x,metadata,names,rows,cache=build_design(data,reg,'screen')
    train=np.flatnonzero(metadata['fold']!=0);valid=np.flatnonzero(metadata['fold']==0)
    altered=permute_training_labels_within_time(metadata,train,policy['seed'])
    np.testing.assert_array_equal(altered['target'][valid],metadata['target'][valid])
    changed=int(np.count_nonzero(altered['target'][train]!=metadata['target'][train]))
    assert changed>0
    mass=[]
    for t in np.unique(metadata['time_online'][train]):
        mask=train[metadata['time_online'][train]==t]
        assert altered['target'][mask].sum()==metadata['target'][mask].sum()
        mass.append([int(t),len(mask),int(metadata['target'][mask].sum())])
    atomic_json(out/'FIT_START.json',{'created_utc':utc_now(),'new_fits':1,'permutation_seed':policy['seed'],
        'changed_training_targets':changed,'training_target_sha256':hashlib.sha256(altered['target'][train].tobytes()).hexdigest(),
        'validation_target_sha256':hashlib.sha256(altered['target'][valid].tobytes()).hexdigest()},immutable=True)
    progress('Limited label-permutation falsification','One B2 screen-fold-0 control fit; target shuffling stays inside training time strata')
    model,expert,training=fit_base(x,altered,train,valid,guard=data.guard,registration=reg)
    model.save(out/'permuted_control.joblib')
    refs,proof=fold_predictions(data,0,'screen',rows[valid]);prediction=combine(expert,refs['C205'],.8)
    original=BASE/'experiments'/candidate/'screen';r=read_json(original/'fold_0.json')
    for filename in ('fold_0.npy','fold_0_expert.npy'):
        assert digest(original/filename)==r['files_sha256'][filename]
    real_expert=np.load(original/'fold_0_expert.npy',allow_pickle=False)
    target=metadata['target'][valid];times=metadata['time_online'][valid];ids=metadata['dataset_id'][valid]
    code,counts=draw_group_multiplicities(ids,2000,policy['seed']);n=int(code.max()+1)
    a=PreparedAUC.create(target,real_expert,times,code,n).evaluate(counts)
    b=PreparedAUC.create(target,expert,times,code,n).evaluate(counts)
    constant=np.full(len(expert),.5,dtype=np.float32);constant_score=ts_auc(target,constant,times)
    assert constant_score==.5
    invalid=detect_invalid_future_controls(Online,np.arange(10,dtype=np.float32));assert len(invalid)==3 and all(invalid.values())
    np.save(out/'control_predictions.npy',prediction,allow_pickle=False)
    np.save(out/'control_expert.npy',expert,allow_pickle=False)
    np.save(out/'paired_expert_bootstrap.npy',np.column_stack([a,b]),allow_pickle=False)
    data.guard.check_state()
    files=['permuted_control.joblib','control_predictions.npy','control_expert.npy','paired_expert_bootstrap.npy']
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'candidate':'CONTROL_LABEL_PERMUTATION_B2_SCREEN_FOLD0',
        'not_a_selection_candidate':True,'new_candidate_identity':False,'new_fits':1,'frozen_base_new_fits':0,
        'training':training,'training_targets_changed':changed,'validation_targets_unchanged':True,
        'training_time_class_counts_preserved':True,'permutation_seed':policy['seed'],'seal_rows':0,
        'constant_prediction_TS_AUC':constant_score,'invalid_future_controls_detected':invalid,
        'original_expert_TS_AUC':ts_auc(target,real_expert,times),'permuted_expert_TS_AUC':ts_auc(target,expert,times),
        'permuted_fixed_composite_TS_AUC':ts_auc(target,prediction,times),'reference_BL80_TS_AUC':ts_auc(target,refs['BL80'],times),
        'paired_permuted_minus_original_expert':paired_summary(a[:,None],b[:,None]),
        'control_expectation':'Destroy train feature/target association while preserving train time-class pair mass. Exact score is reported without searching permutations or asserting a calibrated chance test from one fit.',
        'policy_sha256':digest(policy_path),'frozen_sources_sha256':proof,'files_sha256':{name:digest(out/name) for name in files}},immutable=True)
    print('NEXT4 LIMITED NEGATIVE CONTROLS COMPLETE',flush=True)


if __name__=='__main__':main()
