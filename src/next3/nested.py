"""Disjoint grouped lineage for gate-training base OOF inside outer training."""
import hashlib
import numpy as np


def array_hash(values):
    array=np.ascontiguousarray(values)
    return hashlib.sha256(str(array.dtype).encode()+str(array.shape).encode()+array.tobytes()).hexdigest()


def nested_partitions(metadata, outer_train, outer_valid, guard):
    train=np.asarray(outer_train,dtype=np.int64)
    valid=np.asarray(outer_valid,dtype=np.int64)
    if len(np.unique(train))!=len(train) or len(np.unique(valid))!=len(valid) or np.intersect1d(train,valid).size:
        raise ValueError('Outer coordinate sets must be unique and disjoint')
    guard.partition(np.unique(metadata['dataset_id'][train]),np.unique(metadata['dataset_id'][valid]))
    folds=np.unique(metadata['fold'][train])
    if len(folds)<3:
        raise ValueError('At least three pre-existing grouped inner partitions required')
    coverage=np.zeros(len(train),dtype=np.int8)
    result=[]
    for fold in folds:
        mask=metadata['fold'][train]==fold
        inner_train=train[~mask];inner_valid=train[mask]
        guard.partition(np.unique(metadata['dataset_id'][inner_train]),np.unique(metadata['dataset_id'][inner_valid]))
        if np.intersect1d(inner_train,valid).size or np.intersect1d(inner_valid,valid).size:
            raise RuntimeError('Outer validation leaked into nested base OOF')
        coverage[mask]+=1
        result.append((int(fold),inner_train,inner_valid))
    if not np.all(coverage==1):
        raise RuntimeError('Inner OOF does not cover outer training exactly once')
    return result
