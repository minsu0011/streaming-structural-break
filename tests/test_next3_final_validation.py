import datetime as dt
import os
import pytest
from src.next3.final_validation import all_true, final_clock, tests_proof as audit_tests_proof


@pytest.mark.parametrize('checks', [{}, {'a': 1}, {'a': 'PASS'}, {'a': None}, {'a': False}])
def test_final_checks_do_not_accept_empty_or_truthy_non_boolean_receipts(checks):
    with pytest.raises((RuntimeError, ValueError)): all_true(checks)


def test_final_checks_accept_only_successful_literal_checks():
    all_true({'source': True, 'raw_training': True})


def session(audit_seconds=40200):
    started = dt.datetime(2026, 9, 10, tzinfo=dt.timezone.utc)
    audit = started+dt.timedelta(seconds=audit_seconds)
    return {'started_utc': started.isoformat(), 'audit_started_utc': audit.isoformat()}, audit


def test_final_closure_requires_separate_audit_duration():
    record, audit = session()
    with pytest.raises(RuntimeError): final_clock(record, audit+dt.timedelta(seconds=2999.9), require_complete_duration=True)
    result = final_clock(record, audit+dt.timedelta(seconds=3000), require_complete_duration=True)
    assert result['wall_seconds'] == 43200 and result['research_wall_seconds'] == 40200


@pytest.mark.parametrize('mode', ['early_audit', 'future_audit', 'naive_clock'])
def test_final_clock_rejects_invalid_phase_evidence(mode):
    record, audit = session(40199 if mode == 'early_audit' else 40200)
    now = audit-dt.timedelta(seconds=1) if mode == 'future_audit' else audit+dt.timedelta(seconds=4000)
    if mode == 'naive_clock': now = now.replace(tzinfo=None)
    with pytest.raises((RuntimeError, ValueError)): final_clock(record, now)


def xml(tmp_path, body='', counters='', classname='tests.test_ts_auc'):
    path = tmp_path/'tests.xml'
    path.write_text(f'<testsuites><testsuite tests="1" time=".2" {counters}>'
                    f'<testcase classname="{classname}" name="oracle">{body}</testcase></testsuite></testsuites>', encoding='utf-8')
    return path


def test_final_xml_verifies_actual_cases_and_required_scorer(tmp_path):
    result = audit_tests_proof(xml(tmp_path), minimum_count=1, required_classnames={'tests.test_ts_auc'}, not_before_timestamp=0)
    assert result['tests'] == 1 and result['seconds'] == .2


@pytest.mark.parametrize('mode', ['nested_failure', 'skip_counter', 'missing_scorer', 'stale', 'incomplete'])
def test_final_xml_rejects_plausible_but_incomplete_success(tmp_path, mode):
    path = xml(tmp_path, body='<failure>actual failure</failure>' if mode == 'nested_failure' else '',
        counters='skipped="1"' if mode == 'skip_counter' else '',
        classname='tests.test_other' if mode == 'missing_scorer' else 'tests.test_ts_auc')
    if mode == 'stale': os.utime(path, (1, 1))
    with pytest.raises(RuntimeError):
        audit_tests_proof(path, minimum_count=2 if mode == 'incomplete' else 1,
            required_classnames={'tests.test_ts_auc'}, not_before_timestamp=2 if mode == 'stale' else 0)
