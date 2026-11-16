import src.next8
from types import SimpleNamespace
import numpy as np
from src.next8.common import ROOT,read_json
from src.next8.data import FitScope
from src.next8.early_reference import fit_early_reference
from src.next5.model import feature_module


def test_early_reference_cannot_read_late_or_validation_labels():
    registration=read_json(ROOT/'artifacts/next5/references/registrations/REF_J102.json')
    names=feature_module(registration['feature_module']).feature_names(registration['feature_settings'])
    rng=np.random.default_rng(120);ids=np.repeat(np.arange(8),512);time=np.tile(np.arange(512),8)
    target=rng.integers(0,2,len(ids));x=rng.normal(size=(len(ids),len(names))).astype(np.float32)
    class EarlyTrainOnly:
        def __getitem__(self,rows):
            assert np.all(ids[rows]<4) and np.all(time[rows]<=199)
            return target[rows]
    meta={'dataset_id':ids,'time_online':time,'target':EarlyTrainOnly()}
    scope=FitScope(frozenset(range(4)),frozenset(range(4,8)),frozenset(range(4,8)),'early_fixture')
    class Guard:
        split=SimpleNamespace(digest='synthetic_fixture')
        def partition(self,a,b):assert set(a).isdisjoint(b)
    model=fit_early_reference(x,meta,scope,registration,Guard())
    assert model.training['training_rows']==800 and model.training['training_max_online_time']==199
    assert model.training['native_compact_max_abs_difference']<1e-12
    assert model.estimator.get_params()['n_jobs']==1
    model.validate()
    p=model.predict(x[(ids>=4)&(time>=400)])
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
