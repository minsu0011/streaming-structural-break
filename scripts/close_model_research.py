"""Record evidence-based transition to audit-only work without moving deadlines."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.validation.adaptive import ResearchBudget,select_next
from src.utils.artifacts import write_json,sha256,utc_now,ledger


def main():
    path=ROOT/'artifacts/real/MODEL_RESEARCH_CLOSURE.json'
    if path.exists():raise RuntimeError('Model research closure is immutable')
    state=json.loads((ROOT/'artifacts/real/RESEARCH_STATE.json').read_text());control=json.loads((ROOT/'artifacts/real/CONTROL.json').read_text())
    usage=json.loads((ROOT/'REDUCED_TEST_USAGE_LEDGER.json').read_text());budget=ResearchBudget(ROOT/'artifacts/validation/REAL_SESSION.json')
    if state.get('active_candidate') or not control.get('pause_after_candidate'):raise RuntimeError('A clean paused checkpoint is required')
    if usage['scored_uses']!=1 or any(e['kind']=='primary' for e in usage['entries']):raise RuntimeError('Closure must precede the final reduced diagnostic')
    if not all(r['status']=='COMPLETE' and len(r['fold_scores'])==5 for r in state['results']):raise RuntimeError('Incomplete candidate in final results')
    diagnostic=json.loads((ROOT/'artifacts/real/arch_historical_residual_audit/RESULTS.json').read_text())
    if diagnostic['status']!='PASS' or diagnostic['ARCH2_followup_trigger_met']:raise RuntimeError('The last bounded mechanism audit did not close its proposed follow-up')
    next_candidate=select_next(state['results'],set(state['attempted']),state['primary_id'])
    if next_candidate is not None:raise RuntimeError('The bounded adaptive queue still has an unattempted candidate')
    report={'status':'CLOSED_FOR_AUDIT_ONLY','timestamp':utc_now(),'completed_candidates':len(state['results']),'primary_id':state['primary_id'],
        'research_state_sha256':sha256(ROOT/'artifacts/real/RESEARCH_STATE.json'),'adaptive_queue_exhausted':True,
        'reason':'Recent fixed historical quantile and magnitude hypotheses failed promotion; the next variance-lag mechanism failed its prospectively recorded historical admission rule. Continue substantive audit/reproduction instead of adding an unmotivated sweep.',
        'deadline_utc':budget.state['deadline_utc'],'new_experiment_deadline_utc':budget.state['new_experiment_deadline_utc'],
        'final_reduced_diagnostic_not_before_utc':budget.state['new_experiment_deadline_utc'],'score_gates_changed':False,
        'LOCAL_FINAL_SEAL_OPENED':False,'reduced_scores_consulted':False,'reduced_scored_uses':usage['scored_uses']}
    write_json(path,report)
    budget.checkpoint(status='AUDIT_ONLY_PREPARATION',model_research_closed=True,model_research_closure_sha256=sha256(path),
        next_action='Freeze and refit DEV candidates, audit actual final artifacts and reproduce source. The final reduced diagnostic remains delayed until the original final buffer.')
    ledger(experiment_id='model_research_closure',model=state['primary_id'],status='AUDIT_ONLY_TRANSITION',notes='249 complete hypotheses; original deadlines and score gates preserved; no further model selection after final freeze, reduced diagnostic delayed.')
    print('MODEL RESEARCH CLOSED',len(state['results']),'candidates; original final deadline preserved',flush=True)


if __name__=='__main__':main()
