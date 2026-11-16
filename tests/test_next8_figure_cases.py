"""Case picking is first eligible, includes adverse cases, and never tunes."""
import src.next8
import numpy as np
from src.next8.figure_cases import choose_case_indices


def test_first_eligible_success_failure_and_missed_event():
    ids=np.repeat([31,32,33],40);t=np.tile(np.arange(40)*8,3)
    delta=np.ones(120);pred=np.zeros(120);pred[:40]=.5;pred[40:80]=-1.;pred[80:]=1.
    reversal=np.zeros(120,dtype=int);reversal[2]=1
    probability=np.zeros(120);probability[4]=.75
    result,counts=choose_case_indices(ids,t,{'delta':delta,'reversal':reversal},pred,probability)
    assert result['success']['series_id']==31 and counts['success']==2
    assert result['failure']['series_id']==32 and counts['failure']==1
    assert result['delay_or_miss']['first_later_forecast_time']==32 and result['delay_or_miss']['miss_time']==16
    assert result['false_alarm']['false_forecast_time']==32


def test_unavailable_categories_are_not_created_by_relaxing_cutoffs():
    ids=np.zeros(16,dtype=int);t=np.arange(16)
    result,counts=choose_case_indices(ids,t,{'delta':np.zeros(16),'reversal':np.zeros(16)},np.zeros(16),np.zeros(16))
    assert result=={} and not any(counts.values())
