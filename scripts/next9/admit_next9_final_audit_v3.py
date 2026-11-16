"""Admit final audit only after actual10h research and all research work closes."""
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,utc_now,activity

def main():
    from src.next9.work_time import snapshot
    from src.next9.queue import identity_alive
    import psutil
    clock=snapshot();assert clock['productive_seconds']>=36000,clock['productive_seconds']
    for item in read_json(BASE/'runtime/TASK_QUEUE.json'):
        r=read_json(BASE/'runtime/queue'/item['id']/'TERMINAL.json')
        assert r['status']=='COMPLETE' or (item['id']=='ORIENTATION_V2_outer_4_inner_0' and r['status']=='QUARANTINE')
    for path in (BASE/'runtime/queue').glob('*/attempt_*/START.json'):
        r=read_json(path);assert not identity_alive(r['pid'],r['process_created']),str(path)
    for name in ['JOINT_SOLVER_AUDIT_NEXT9.json','SCIENTIFIC_RESULTS_NEXT9.json','SCIENTIFIC_REPORTS_DONE_NEXT9.json','PUBLIC_ENTRY_REPLAY_NEXT9.json',
        'STATE_OBSERVABILITY_NEXT9.json','LATENCY_NEXT9.json']:
        assert read_json(BASE/name)['status'] in ['PASS','COMPLETE','FAIL'],name
    assert read_json(BASE/'JOINT_SOLVER_AUDIT_NEXT9.json')['status']=='PASS'
    visual=read_json(BASE/'FIGURES/ALL_TEN_VISUALLY_INSPECTED_NEXT9.json');assert visual['status']=='PASS'
    test=read_json(BASE/'tests/FINAL_TEST_SUITE_NEXT9.json');assert test['status']=='PASS'
    for name,sha in test['source_sha256'].items():assert digest(ROOT/name)==sha,name
    owner=read_json(BASE/'runtime/SCHEDULER_OWNER.json')
    if identity_alive(owner['pid'],owner['process_created']):
        process=psutil.Process(owner['pid'])
        assert any(str(x).endswith('run_next9_autonomous.py') for x in process.cmdline())
        atomic(BASE/'runtime/STOP_SCHEDULER.json',dict(status='STOP_REQUESTED',timestamp=utc_now(),
            reason='All registered research complete,10hproductive fulfilled, admission to separate finalaudit',owned_pid=owner['pid']))
        process.wait(timeout=45)  # Outside every credited activity interval.
    assert not identity_alive(owner['pid'],owner['process_created'])
    with activity('Close research sources after owned idle scheduler has stopped','RESEARCH_CLOSURE','HANDOFF'):
        atomic(BASE/'runtime/SCHEDULER_STOPPED_NEXT9.json',dict(status='PASS',timestamp=utc_now(),owner=owner,normal_stop=True,no_owned_queue_workers_alive=True))
        sources={}
        for directory in [ROOT/'src/next9',ROOT/'scripts/next9']:
            for path in directory.rglob('*'):
                if path.is_file() and path.suffix in ['.py','.ini'] and '__pycache__' not in path.parts:sources[path.relative_to(ROOT).as_posix()]=digest(path)
        atomic(BASE/'runtime/FINAL_SOURCE_CLOSURE_NEXT9.json',dict(status='FROZEN',timestamp=utc_now(),source_sha256=sources,
            test_receipt_sha256=digest(BASE/'tests/FINAL_TEST_SUITE_NEXT9.json'),new_fits_or_source_changes_after_closure_forbidden=True))
    clock=snapshot();assert clock['productive_seconds']>=36000
    # Must be after the closure activity ended, never split a research interval.
    atomic(BASE/'runtime/FINAL_AUDIT_ADMISSION_NEXT9.json',dict(status='PASS',timestamp=utc_now(),admitted_epoch=time.time(),
        productive_seconds=clock['productive_seconds'],minimum_productive_seconds=36000,minimum_separate_audit_seconds=3600,
        all_registered_research_complete=True,owned_scheduler_stopped=True,waiting_or_sleep_not_credited=True,
        source_closure_sha256=digest(BASE/'runtime/FINAL_SOURCE_CLOSURE_NEXT9.json')))

if __name__=='__main__':main()
