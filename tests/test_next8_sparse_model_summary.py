"""Even early all-positive evaluation stays explicit rather than fabricated."""
import src.next8
import numpy as np
from src.next8.evaluation import METHODS


def test_missing_early_primary_does_not_average_partial_folds(monkeypatch):
    import src.next8.model_summary as module
    def load(fold,name,early):
        r={'points':{m:{'TS_AUC':None,'pairs':0} for m in METHODS},
            'prediction_provenance':{'integration_eligible':False,'integration_status':'NOT_RUN','prediction_sha256':'fixture'}}
        return r,{m:np.full(2000,np.nan) for m in METHODS}
    monkeypatch.setattr(module,'load_domain',load)
    monkeypatch.setattr(module,'read_json',lambda p:{'domains':['ALL']})
    result=module.summarize_models(True)
    assert all(v is None for v in result['mean_TS_AUC'].values())
    assert all(not r['historical_numeric_promotion_gate'] for r in result['comparisons'])
    assert result['simultaneous_family_envelope']['status']=='INSUFFICIENT_SUPPORT'
