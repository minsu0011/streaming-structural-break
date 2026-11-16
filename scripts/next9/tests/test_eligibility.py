import copy
import pytest
from src.next9.eligibility import holm,panel_pass,mode_pass,select


def sample(feature=0):
    rows=[dict(horizon=h,groups=100,H_center_MSE=1.,
               persistence_contrast=dict(mean_gain=.1,holm_adjusted_p=.01),
               H_center_contrast=dict(mean_gain=.1,holm_adjusted_p=.01)) for h in [1,4,16]]
    return dict(feature=feature,panels={m:[copy.deepcopy(rows) for _ in range(3)] for m in ['NORMAL','EARLY_LATE','EARLY_SELECT']},
                pooled={m:copy.deepcopy(rows) for m in ['NORMAL','EARLY_LATE','EARLY_SELECT']})


def test_single_best_horizon_cannot_select_a_feature():
    r=sample();rows=r['pooled']['NORMAL']
    rows[1]['H_center_contrast']['mean_gain']=-.01
    assert not mode_pass(r['panels']['NORMAL'],rows)
    with pytest.raises(ValueError):panel_pass(rows[:1])


def test_early_selection_does_not_use_late_outcomes():
    r=sample()
    assert select([r],early_only=True)[0]==[0]
    r['pooled']['EARLY_LATE'][0]['H_center_contrast']['mean_gain']=-100
    assert select([r],early_only=True)[0]==[0]
    assert select([r],early_only=False)[0]==[]


def test_feature_tie_break_is_id_not_break_score():
    records=[sample(i) for i in range(20,0,-1)]
    selected,total=select(records)
    assert total==20
    assert selected==list(range(1,13))


def test_multiple_testing_family_includes_all_features():
    assert holm({'a':.01,'b':.04,'c':.9})=={'a':.03,'b':.08,'c':.9}
