"""Meaningful clock, ownership, dependency and registration boundary checks."""
from pathlib import Path
import os
import psutil
import pytest
from src.next4 import scheduler as s


@pytest.mark.parametrize('child,recent,kind,expected',[
    (True,False,True,20),(False,True,True,20),(False,False,True,0),
    (True,False,False,0),(False,True,False,0)])
def test_only_observed_substantive_work_counts(child,recent,kind,expected):
    r=s.observe_interval(20,child_progress=child,recent_activity=recent,work_is_research=kind,end_wall=100)
    assert r['research_credit_seconds']==expected
    assert r['research_credit_seconds']+r['unobserved_seconds']==20


def test_long_observation_gap_not_backcredited():
    r=s.observe_interval(121,child_progress=True,recent_activity=True,work_is_research=True,end_wall=1000)
    assert r['research_credit_seconds']==0 and r['unobserved_seconds']==121


def test_audit_interval_split_at_actual_transition():
    r=s.observe_interval(20,child_progress=True,recent_activity=False,work_is_research=True,end_wall=100,audit_started_wall=91)
    assert r['research_credit_seconds']==11 and r['audit_credit_seconds']==9


@pytest.mark.parametrize('delta',[-1,float('nan'),float('inf')])
def test_invalid_clock_rejected(delta):
    with pytest.raises(ValueError):s.observe_interval(delta,child_progress=True,recent_activity=True,work_is_research=True,end_wall=100)


def test_pid_identity_requires_creation_time():
    p=psutil.Process(os.getpid())
    assert s.process_matches(p.pid,p.create_time())
    assert not s.process_matches(p.pid,p.create_time()-1)


@pytest.fixture
def queue(tmp_path,monkeypatch):
    root=tmp_path/'root';(root/'scripts').mkdir(parents=True);(root/'scripts/task.py').write_text('pass\n')
    base=root/'artifacts/next4';monkeypatch.setattr(s,'ROOT',root);monkeypatch.setattr(s,'BASE',base)
    return root,base


def job(name='P001'):
    return {'id':name,'kind':'preflight','argv':['scripts/task.py'],'description':'Small registered task'}


def test_dependency_must_be_registered_and_successful(queue):
    _,base=queue
    with pytest.raises(ValueError):s.register_job({**job('P002'),'depends_on_success':['P001']})
    s.register_job(job());second={**job('P002'),'depends_on_success':['P001']};s.register_job(second)
    assert not s.prerequisites_succeeded(second,base)
    s.atomic_json(base/'jobs/P001/STATE.json',{'id':'P001','status':'COMPLETE','exit_code':False})
    assert not s.prerequisites_succeeded(second,base)
    s.atomic_json(base/'jobs/P001/STATE.json',{'id':'P001','status':'COMPLETE','exit_code':0})
    assert s.prerequisites_succeeded(second,base)


@pytest.mark.parametrize('change',[{'id':'../bad'},{'argv':'scripts/task.py'},{'argv':['-m','pip']},
    {'argv':['../escape.py']},{'kind':'unknown'},{'kind':'research','candidate_id':'UNREGISTERED'},
    {'depends_on':['P000']},{'dependencies':[]},{'requires':[]},{'priority':True},{'priority':-1}])
def test_unregistered_or_unsafe_job_rejected(queue,change):
    with pytest.raises(ValueError):s.register_job({**job(),**change})


def test_registration_is_immutable(queue):
    s.register_job(job())
    with pytest.raises(FileExistsError):s.register_job(job())
