"""A failed worker must not release a downstream statistical job."""
import pytest
from scripts import advance_next4_base_stages as stage
from src.next4 import scheduler as scheduler
from src.next4.artifacts import atomic_json,read_json


@pytest.mark.parametrize('screen_pass',[True,False])
def test_stage_advancement_obeys_screen_and_successful_worker_dependency(tmp_path,monkeypatch,screen_pass):
    root=tmp_path/'root';base=root/'artifacts/next4';(root/'scripts').mkdir(parents=True)
    for filename in ('evaluate_next4_base.py','bootstrap_next4_candidate.py'):
        (root/'scripts'/filename).write_text('pass\n')
    monkeypatch.setattr(stage,'BASE',base);monkeypatch.setattr(scheduler,'BASE',base);monkeypatch.setattr(scheduler,'ROOT',root)
    atomic_json(base/'families/A_MENU_LOCK.json',{'family':'A','candidate_ids':['A_TEST']})
    atomic_json(base/'registrations/A_TEST.json',{'created_before_score':True})
    atomic_json(base/'experiments/A_TEST/screen/RESULTS.json',{'screen_pass':screen_pass})
    atomic_json(base/'queue/A001_SCREEN.json',{'id':'A001_SCREEN'})
    stage.main()
    if not screen_pass:
        assert not (base/'queue/A101_FULL.json').exists()
        assert not (base/'queue/A201_BOOTSTRAP.json').exists()
        return
    full=read_json(base/'queue/A101_FULL.json');bootstrap=read_json(base/'queue/A201_BOOTSTRAP.json')
    assert not scheduler.prerequisites_succeeded(full,base)
    assert not scheduler.prerequisites_succeeded(bootstrap,base)
    atomic_json(base/'jobs/A001_SCREEN/STATE.json',{'id':'A001_SCREEN','status':'COMPLETE','exit_code':0})
    assert scheduler.prerequisites_succeeded(full,base)
    for status,code,expected in [('RUNNING',None,False),('FAILED',1,False),('QUARANTINED',1,False),('COMPLETE',0,True)]:
        atomic_json(base/'jobs/A101_FULL/STATE.json',{'id':'A101_FULL','status':status,'exit_code':code})
        assert scheduler.prerequisites_succeeded(bootstrap,base) is expected
    stage.main()
    assert len(list((base/'queue').glob('*.json')))==3
