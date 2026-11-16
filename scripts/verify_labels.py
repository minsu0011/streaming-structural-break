"""Empirical tau alignment using official target parquet; no prediction scoring."""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.data.labels import labels_from_tau
from src.utils.artifacts import write_json

def check_pair(index_path,target_path,ids=None):
    filters=[('id','in',list(ids))] if ids is not None else None
    tau=pd.read_parquet(index_path,filters=filters)
    target=pd.read_parquet(target_path,filters=filters).sort_index()
    if not target.index.is_unique or list(target.index.names)!=['id','time']:
        raise ValueError('Invalid target index')
    checked=0; breaks=0
    for sid,part in target.groupby(level='id',sort=False):
        expected=labels_from_tau(len(part),tau.loc[sid,'tau_index'])
        if not np.array_equal(expected,part.target.to_numpy()):
            raise AssertionError(f'Official label mismatch at series {sid}')
        checked+=len(part); breaks+=int(expected.any())
    if set(tau.index)!=set(target.index.get_level_values('id')): raise ValueError('Target ID mismatch')
    return {'series':len(tau),'rows':checked,'break_series':breaks,'mismatch_rows':0}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--data',default='data/raw'); args=parser.parse_args(); directory=Path(args.data)
    from src.validation.splits import FrozenSplit
    split=FrozenSplit(ROOT/'artifacts/validation')
    ids=split.frame.loc[split.frame.role=='DEVELOPMENT_POOL','dataset_id'].tolist()
    results={'development_train':check_pair(directory/'y_train_index.parquet',directory/'y_train.parquet',ids)}
    status='PASS'
    report={'status':status,'convention':'online t >= tau; -1/None means no break','checks':results,'reduced_scored_uses':0,
        'seal_target_rows_selected':0,'reduced_target_rows_selected':0,'purpose':'Development label alignment only; no model predictions'}
    write_json(ROOT/'artifacts/validation/LABEL_ALIGNMENT.json',report)
    print(report)

if __name__=='__main__': main()
