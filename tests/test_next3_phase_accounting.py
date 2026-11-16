import datetime
import math
import pytest
from src.next3.phase_accounting import split_observed_interval
from src.next3 import maintenance
from src.next3.artifacts import atomic_json,read_json
from tests.test_next3_supervisor_lifecycle import tick_fixture


@pytest.mark.parametrize('audit_start,research,audit',[
    (None,45.,0.),(0.,0.,45.),(55.,0.,45.),(70.,15.,30.),
    (99.875,44.875,.125),(100.,45.,0.),(110.,45.,0.)])
def test_observed_interval_is_partitioned_at_actual_boundary(audit_start,research,audit):
    result=split_observed_interval(45.,observed=True,interval_end_wall=100.,audit_started_wall=audit_start)
    assert result['research_credit_seconds']==research
    assert result['audit_credit_seconds']==audit
    assert result['observed_credit_seconds']==research+audit==45.
    assert result['unobserved_seconds']==0.


@pytest.mark.parametrize('audit_start',[None,0.,70.,110.])
def test_unobserved_gap_never_becomes_productive_at_a_phase_change(audit_start):
    result=split_observed_interval(120.,observed=False,interval_end_wall=100.,audit_started_wall=audit_start)
    assert result['research_credit_seconds']==result['audit_credit_seconds']==result['observed_credit_seconds']==0.
    assert result['unobserved_seconds']==120.


def test_zero_interval_has_no_credit_even_at_the_boundary():
    result=split_observed_interval(0.,observed=True,interval_end_wall=100.,audit_started_wall=100.)
    assert all(value==0 for value in result.values())


@pytest.mark.parametrize('delta,observed,end,start',[
    (-1.,True,100.,None),(math.nan,True,100.,None),(math.inf,True,100.,None),
    (45.,1,100.,None),(45.,True,math.inf,None),(45.,True,100.,math.nan)])
def test_invalid_clock_or_activity_inputs_are_rejected(delta,observed,end,start):
    with pytest.raises(ValueError):
        split_observed_interval(delta,observed=observed,interval_end_wall=end,audit_started_wall=start)


@pytest.mark.parametrize('audit_age',[0.,.125,13.5,45.,60.])
@pytest.mark.parametrize('observed',[False,True])
def test_real_supervisor_tick_keeps_existing_totals_and_splits_only_the_current_interval(tmp_path,monkeypatch,audit_age,observed):
    supervisor,_=tick_fixture(tmp_path,monkeypatch,interval=45.,recent=observed,busy=False)
    end=datetime.datetime.now(datetime.timezone.utc).timestamp()
    session=read_json(tmp_path/'SESSION.json')
    session['audit_started_utc']=datetime.datetime.fromtimestamp(end-audit_age,datetime.timezone.utc).isoformat()
    atomic_json(tmp_path/'SESSION.json',session)
    monkeypatch.setattr(maintenance.time,'time',lambda:end)
    supervisor.tick();report=read_json(tmp_path/'runtime/ACCOUNTING.json')
    expected_audit=min(45.,audit_age) if observed else 0.
    expected_research=(45. if observed else 0.)-expected_audit
    assert report['productive_seconds']==pytest.approx(10.+expected_research,abs=1e-6)
    assert report['audit_productive_seconds']==pytest.approx(expected_audit,abs=1e-6)
    assert report['unobserved_seconds']==(0. if observed else 45.)
    assert report['last_interval_credited']==(45. if observed else 0.)
    assert report['last_interval_research_credit_seconds']+report['last_interval_audit_credit_seconds']==report['last_interval_credited']
