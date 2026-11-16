"""Innovation-group pairing and fixed semantic tensor contrasts."""
import src.next8
import numpy as np
from src.next8.synthetic_summary import bootstrap_group_means,synthetic_summary_tables,MODEL_PAIRS,ERROR_PAIRS
from src.next8.synthetic_evaluation import WINDOWS,SCORE_METHODS,AUX_METHODS
from src.next8.features import DETAILED,HORIZONS,N_FEATURES


def test_matched_innovation_shift_and_duplicate_columns():
    rng=np.random.default_rng(34);null=rng.normal(size=128);intervention=null+.25
    values=np.column_stack((intervention-null,null,null,np.full(128,np.nan)))
    result=bootstrap_group_means(values,np.zeros(128,dtype=int))
    assert abs(result['point'][0]-.25)<1e-15 and result['ci_low'][0]>.249999999999
    assert result['ci_low'][1]==result['ci_low'][2] and result['ci_high'][1]==result['ci_high'][2]
    assert result['exact_alias_columns']==1 and result['groups'][3]==0 and np.isnan(result['point'][3])


def test_sparse_unobserved_H_states_do_not_become_zero_error():
    values=np.full((128,2),np.nan);values[:63,0]=.1;values[:64,1]=.1
    result=bootstrap_group_means(values,np.zeros(128,dtype=int))
    assert result['groups'].tolist()==[63,64]
    assert np.isnan(result['ci_low'][0]) and result['ci_low'][1]==.1


def test_matched_model_and_forecast_contrast_semantics(monkeypatch):
    import src.next8.synthetic_summary as module
    def check(values,definitions,*,kind):
        assert np.prod(values.shape[2:])==len(definitions)
        flat=values[0,0].reshape(-1)
        if kind=='MATCHED_MODEL_SCORE_DIFFERENCE':
            for value,d in zip(flat,definitions):
                assert value==SCORE_METHODS.index(d['candidate'])-SCORE_METHODS.index(d['reference'])
        if kind=='MATCHED_FORECAST_LOSS_GAIN':
            for value,d in zip(flat,definitions):
                assert value==AUX_METHODS.index(d['reference'])-AUX_METHODS.index(d['candidate'])
        if kind=='MATCHED_INTERVENTION_RESPONSE':assert np.all(flat==0)
        return [{'kind':kind}],[]
    monkeypatch.setattr(module,'tensor_rows',check)
    score=np.broadcast_to(np.arange(len(SCORE_METHODS))[None,None,None,None,:,None],(512,2,5,len(WINDOWS),len(SCORE_METHODS),4))
    error=np.broadcast_to(np.arange(len(AUX_METHODS))[None,None,None,None,:,None,None,None],
        (512,2,len(DETAILED),len(HORIZONS),len(AUX_METHODS),5,len(WINDOWS),2))
    clip=np.broadcast_to(0.,(512,2,5,N_FEATURES))
    result=list(synthetic_summary_tables({'score':score,'error':error,'clip':clip}))
    assert len(result)==6
