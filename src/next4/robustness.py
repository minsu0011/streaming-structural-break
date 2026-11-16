"""Original ALT/H assignments and matching frozen component OOF, read-only."""
import numpy as np
import pandas as pd
from src.next4.artifacts import ROOT,read_json,digest
from src.next2.blend import combine


def partitions(data,metadata):
    old=read_json(ROOT/'artifacts/next2/robustness/C205_DEPENDENCE/POLICY.json')
    path=ROOT/'ROBUSTNESS_SPLIT_LOCK.json';assert digest(path)==old['old_split_policy_sha256']
    policy=read_json(path);regime_dir=ROOT/'artifacts/next/current_forensic'
    assert digest(regime_dir/'REGIME_ASSIGNMENT_LOCK.json')==old['regime_lock_sha256']
    lock=read_json(regime_dir/'REGIME_ASSIGNMENT_LOCK.json')
    assert digest(regime_dir/'DEV_SIGNATURES.parquet')==lock['files_sha256']['DEV_SIGNATURES.parquet']
    frame=pd.read_parquet(regime_dir/'DEV_SIGNATURES.parquet',columns=['dataset_id','historical_regime'])
    data.guard.admit(frame.dataset_id.to_numpy(),purpose='NEXT4 existing H-only held-out regime assignment')
    regimes=frame.set_index('dataset_id').historical_regime.to_dict()
    assignments=[(f'ALT_{j+1}',{int(k):v for k,v in item['fold_by_dataset_id'].items()}) for j,item in enumerate(policy['alternates'])]
    assignments.append(('LEAVE_ONE_H_REGIME_OUT',regimes))
    order=np.concatenate([np.flatnonzero(metadata['fold']==fold) for fold in range(5)])
    result={}
    for name,assignment in assignments:
        code=pd.Series(metadata['dataset_id']).map(assignment).to_numpy()
        assert np.isfinite(code).all() and set(np.unique(code))==set(range(5))
        groups=[order[code[order]==fold] for fold in range(5)];result[name]=[]
        for fold in range(5):
            valid=groups[fold];train=np.concatenate([groups[k] for k in range(5) if k!=fold])
            data.guard.partition(np.unique(metadata['dataset_id'][train]),np.unique(metadata['dataset_id'][valid]))
            result[name].append((train,valid))
    return result,{'original_split_lock_sha256':digest(path),'H_regime_lock_sha256':digest(regime_dir/'REGIME_ASSIGNMENT_LOCK.json'),
        'order':'Stress fold, then original fold, then dataset ID and online time, matching original component fits',
        'new_assignments':False,'H_assignment_outcome_blind':True}


def frozen_predictions(data,split,fold,expected_rows):
    if split not in ('ALT_1','ALT_2','ALT_3','LEAVE_ONE_H_REGIME_OUT') or fold not in range(5):raise ValueError('Existing stress split required')
    values={};sources={}
    for role,directory in [('J102',ROOT/'artifacts/next/robustness/candidates/J102_SERIAL_VARIANCE_512'/split),
        ('C205',ROOT/'artifacts/next2/robustness/C205_DEPENDENCE'/split)]:
        receipt=read_json(directory/f'fold_{fold}.json');assert receipt['status']=='PASS'
        for filename in (f'fold_{fold}.npy',f'fold_{fold}_global_rows.npy'):
            assert digest(directory/filename)==receipt['files_sha256'][filename]
            sources[(directory/filename).relative_to(ROOT).as_posix()]=digest(directory/filename)
        np.testing.assert_array_equal(np.load(directory/f'fold_{fold}_global_rows.npy',allow_pickle=False),expected_rows)
        values[role]=np.load(directory/f'fold_{fold}.npy',allow_pickle=False)
    values['BL80']=combine(values['J102'],values['C205'],.8);data.guard.check_state()
    return values,sources
