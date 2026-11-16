import pytest
from src.next5.scheduler import observe_interval,RESEARCH_SECONDS,AUDIT_SECONDS


def test_next5_long_duration_is_explicit():
    assert RESEARCH_SECONDS==36000 and AUDIT_SECONDS==3000


def test_unobserved_gap_is_not_productive():
    r=observe_interval(90,child_progress=True,recent_activity=True,work_is_research=True,end_wall=1000)
    assert r['research_credit_seconds']==0 and r['unobserved_seconds']==90


def test_sleep_and_nonresearch_work_do_not_satisfy_research_minimum():
    for active,research in [(False,True),(True,False)]:
        r=observe_interval(20,child_progress=active,recent_activity=False,work_is_research=research,end_wall=1000)
        assert r['research_credit_seconds']==0


def test_actual_audit_boundary_splits_without_backdating():
    r=observe_interval(20,child_progress=True,recent_activity=False,work_is_research=True,end_wall=1000,audit_started_wall=993)
    assert r['research_credit_seconds']==13 and r['audit_credit_seconds']==7
