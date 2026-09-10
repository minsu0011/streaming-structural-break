import numpy as np
import pandas as pd
import pytest
from src.data.loader import series_frames,load_training
from scripts.verify_labels import check_pair

def make_data(directory):
    index=pd.MultiIndex.from_tuples([(s,t) for s in range(4) for t in range(7)],names=['id','time'])
    data=pd.DataFrame({'value':np.arange(28,dtype=np.float32),'period':np.tile([1,1,1,2,2,2,2],4)},index=index)
    data.to_parquet(directory/'X_train.parquet')
    tau=pd.DataFrame({'tau_index':[-1,0,2,3]},index=pd.Index(range(4),name='id')); tau.to_parquet(directory/'y_train_index.parquet')
    target=data.loc[data.period==2,['value']].rename(columns={'value':'target'})
    target.target=np.concatenate([[0,0,0,0],[1,1,1,1],[0,0,1,1],[0,0,0,1]])
    target.to_parquet(directory/'y_train.parquet')
    return data

@pytest.mark.parametrize('batch_size',[1,3,7,8,11,100])
def test_chunk_boundaries(tmp_path,batch_size):
    data=make_data(tmp_path)
    groups=list(series_frames(tmp_path/'X_train.parquet',batch_size))
    assert [sid for sid,_ in groups]==list(range(4))
    assert pd.concat([part for _,part in groups]).equals(data)
    series=list(load_training(tmp_path,development_only=False,batch_size=batch_size))
    assert [len(x.historical) for x in series]==[3]*4
    assert [x.tau for x in series]==[None,0,2,3]
    assert check_pair(tmp_path/'y_train_index.parquet',tmp_path/'y_train.parquet')['rows']==16

def test_loader_rejects_disorder(tmp_path):
    data=make_data(tmp_path)
    data.iloc[::-1].to_parquet(tmp_path/'X_train.parquet')
    with pytest.raises(ValueError): list(series_frames(tmp_path/'X_train.parquet',5))
