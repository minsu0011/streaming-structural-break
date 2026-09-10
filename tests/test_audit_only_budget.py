from datetime import datetime,timezone
from src.validation.adaptive import ResearchBudget


def test_audit_transition_stops_models_without_shortening_original_deadline(tmp_path):
    start='2026-09-08T06:49:52+00:00';now=datetime(2026,9,8,14,0,tzinfo=timezone.utc)
    path=tmp_path/'budget.json';budget=ResearchBudget(path,start=start)
    assert budget.can_start(1000,now=now)
    deadlines={k:budget.state[k] for k in ('start_utc','deadline_utc','new_experiment_deadline_utc')}
    budget.checkpoint(model_research_closed=True,status='AUDIT_ONLY_PREPARATION')
    restored=ResearchBudget(path)
    assert not restored.can_start(0,now=now)
    assert {k:restored.state[k] for k in deadlines}==deadlines
    assert restored.remaining_experiment_seconds(now)>0
