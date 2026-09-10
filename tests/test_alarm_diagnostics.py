import numpy as np
from src.validation.alarm_diagnostics import training_threshold,first_alarm_outcome,summarize_outcomes


def test_training_quantile_and_strict_ties_bound_empirical_alarm_fraction():
    values=np.array([0.]*75+[.5]*20+[.9]*5,dtype=np.float32)
    for alpha in (.01,.05):
        threshold=training_threshold(values,alpha)
        assert np.mean(values>threshold)<=alpha
        assert not first_alarm_outcome([threshold],None,threshold)['any_alarm']


def test_premature_alarms_and_short_observation_are_not_successful_detection():
    rows=[first_alarm_outcome([.1,.8,.2,.9],2,.5),first_alarm_outcome([.1,.2,.8,.9],2,.5),
        first_alarm_outcome([.1,.2,.1],2,.5),first_alarm_outcome([.8,.1],None,.5),first_alarm_outcome([.1,.2],None,.5)]
    report=summarize_outcomes(rows)
    assert report['null_any_alarm_rate']==.5 and report['break_premature_alarm_count']==1
    assert report['valid_detected_break_count']==1 and report['detected_only_median_delay']==0
    assert report['delay_le_0_eligible_break_series']==3 and report['delay_le_0_valid_detections']==1
    assert report['delay_le_4_eligible_break_series']==0 and report['delay_le_4_rate'] is None
