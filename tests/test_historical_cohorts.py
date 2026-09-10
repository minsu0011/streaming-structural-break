import numpy as np
import pandas as pd
from scripts.diagnose_historical_cohorts import cohort_codes,BINS


def test_boundaries_enter_upper_bin_and_targets_are_ignored():
    frame=pd.DataFrame({key:[low-1,low,high,high+1] for key,(low,high) in BINS.items()})
    frame['target']=[0,0,1,1];frame['tau_index']=[1,2,3,4]
    before=cohort_codes(frame)
    frame['target']=1-frame.target;frame['tau_index']=-100
    after=cohort_codes(frame)
    for key in BINS:
        np.testing.assert_array_equal(before[key],[0,1,2,2])
        np.testing.assert_array_equal(after[key],before[key])
