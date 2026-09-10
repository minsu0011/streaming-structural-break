import numpy as np
import pytest
from src.next.alarm_calibration import null_maximum_threshold,series_maxima,first_alarm_records,summarize_alarms


def test_finite_sample_threshold_and_strict_ties():
    result = null_maximum_threshold(np.arange(99,dtype=np.float32)/100,.05)
    assert result['one_based_order_statistic']==95
    assert result['threshold']==float(np.float32(.94))
    tied = null_maximum_threshold(np.full(99,.5),.05)
    assert tied['calibration_alarm_fraction']==0.
    assert null_maximum_threshold(np.arange(10),.01)['no_finite_threshold']


def test_first_alarm_premature_and_censoring_denominators():
    ids = np.repeat([1,2,3,4],10)
    t = np.tile(np.arange(10),4)
    p = np.zeros(40,dtype=np.float32)
    p[[3,11,16,26]] = .8
    frame = first_alarm_records(ids,t,p,{1:-1,2:5,3:5,4:8},.7)
    result = summarize_alarms(frame)
    assert result['null_false_alarms']==1
    assert result['premature_break_alarms']==1
    assert result['clean_detections']==1
    assert result['conditional_detected_delay_median']==1.
    assert result['eligible_for_5_point_window']==2
    assert result['clean_detection_within_5_points']==.5
    assert result['eligible_for_10_point_window']==0
    assert result['clean_detection_within_10_points'] is None


def test_no_finite_threshold_never_alarms_and_invalid_blocks_rejected():
    frame = first_alarm_records([1,1],[0,1],[1.,1.],{1:-1},None)
    assert not frame.null_false_alarm.any()
    with pytest.raises(ValueError,match='contiguous'):
        series_maxima([1,2,1],[.1,.2,.3])
    with pytest.raises(AssertionError):
        first_alarm_records([1,1],[0,2],[.1,.2],{1:-1},.5)
