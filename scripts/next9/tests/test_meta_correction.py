import src.next9
import numpy as np
import pytest
from src.next9.data import FitScope
from src.next9.meta_correction import validate_oof,fit_correction,fixed_diagnostic


def test_crossfit_meta():
    outer=FitScope(frozenset(range(6)),frozenset([6,7]),frozenset([6,7]),'outer')
    parts=[]
    for k in range(3):
        val=frozenset([2*k,2*k+1]);scope=FitScope(outer.training_ids-val,val,outer.validation_ids,f'inner{k}')
        ids=np.array(sorted(val))
        parts.append(dict(scope=scope,dataset_id=ids,global_row=ids,prediction=np.full(2,.5),
            training_provenance=scope.provenance(),complete_selection_excluded=True))
    np.testing.assert_array_equal(validate_oof(outer,parts,np.arange(6)),.5)
    bad=[dict(v) for v in parts];bad[0]['complete_selection_excluded']=False
    with pytest.raises(ValueError,match='selection'):validate_oof(outer,bad,np.arange(6))
    bad=[dict(v) for v in parts];bad[0]['training_provenance']=outer.provenance()
    with pytest.raises(ValueError,match='wrong-scope'):validate_oof(outer,bad,np.arange(6))


def test_meta_fixed_offset_and_failed_gate():
    rng=np.random.default_rng(9101604);ids=np.arange(128);b=rng.uniform(.1,.9,128);m=rng.uniform(.1,.9,128)
    y=rng.binomial(1,.7*b+.3*m)
    scope=FitScope(frozenset(ids),frozenset([128]),frozenset(),'toy')
    with pytest.raises(ValueError,match='qualify'):fit_correction(b,m,y,ids,ids,scope,dict(status='UNSUPPORTED'))
    gate=dict(status='SUPPORTED',ci_low=.001,positive_folds=3,fold_deltas=[.002]*3)
    model=fit_correction(b,m,y,ids,ids,scope,gate)
    p=model.predict(b,m)
    assert np.isfinite(p).all() and model.training['baseline_logit_offset_coefficient']==1
    assert model.training['learned_M7_quality_not_evaluated_on_its_meta_training_labels']


def test_constant_inner_head_cannot_change_within_partition_BL_rank():
    from src.scoring.ts_auc import ts_auc
    b=np.array([.1,.2,.8,.4,.7,.9]);m=np.full(6,.31);y=np.array([0,1,1,0,0,1]);t=np.zeros(6)
    assert ts_auc(y,b,t)==ts_auc(y,fixed_diagnostic(b,m),t)
