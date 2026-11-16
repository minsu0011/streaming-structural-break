"""The historical .0012 threshold must not be rounded or silently changed."""
import src.next8
from src.next8.model_summary import numerical_promotion


def test_near_miss_and_all_original_gates_remain_exact():
    row={'status':'SUPPORTED','mean_delta':.0011956500192103591,'ci_low':.0001,'positive_folds':5,'worst_fold_delta':.0001}
    assert not numerical_promotion(row,False)
    row['mean_delta']=.0012
    assert numerical_promotion(row,False)
    assert not numerical_promotion(row,True)
    row['ci_low']=0.
    assert not numerical_promotion(row,False)
    row.update(ci_low=.0001,positive_folds=3)
    assert not numerical_promotion(row,False)
