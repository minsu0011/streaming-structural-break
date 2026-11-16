"""Limited falsification controls, never candidate recipes or model inputs."""
import numpy as np


def permute_training_labels_within_time(metadata,train,seed):
    train=np.asarray(train,dtype=np.int64)
    if train.ndim!=1 or len(np.unique(train))!=len(train) or len(train)==0:
        raise ValueError('Unique nonempty training coordinates required')
    result={k:np.asarray(v).copy() if k=='target' else v for k,v in metadata.items()}
    rng=np.random.default_rng(seed);times=np.asarray(metadata['time_online']);target=np.asarray(metadata['target'])
    order=train[np.argsort(times[train],kind='stable')]
    cuts=np.r_[0,np.flatnonzero(np.diff(times[order]))+1,len(order)]
    for start,end in zip(cuts[:-1],cuts[1:]):
        rows=order[start:end];result['target'][rows]=target[rows[rng.permutation(len(rows))]]
    outside=np.ones(len(target),dtype=bool);outside[train]=False
    np.testing.assert_array_equal(result['target'][outside],target[outside])
    return result


def detect_invalid_future_controls(online_class,points):
    detected={}
    stream=online_class(points)
    try:len(stream)
    except AssertionError:detected['future_length']=True
    stream=online_class(points)
    try:np.asarray(stream)
    except AssertionError:detected['materialized_future_suffix']=True
    stream=online_class(points);iterator=iter(stream)
    next(iterator);next(iterator)  # Deliberately invalid lookahead before first prediction.
    try:assert stream.consumed==1,'Current-point output consumed a future observation'
    except AssertionError:detected['one_step_future_shift']=True
    return detected
