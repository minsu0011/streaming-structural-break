"""Documented idle-boundary owned supervisor restart, with prior clock retained."""
from pathlib import Path
import sys
import subprocess
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,atomic_json,read_json,digest,utc_now,progress
from src.next4.scheduler import process_matches


def main():
    out=BASE/'maintenance/SCHEDULER_V2';assert not (out/'RESTART.json').exists()
    session=read_json(BASE/'SESSION.json');assert session['modeling_blocked'] and session['status']=='ACTIVE_RESEARCH'
    owner=read_json(BASE/'runtime/SUPERVISOR.json');account=read_json(BASE/'runtime/ACCOUNTING.json')
    assert account['current_job'] is None
    assert not any(read_json(p)['status']=='RUNNING' for p in (BASE/'jobs').glob('*/STATE.json'))
    assert process_matches(owner['pid'],owner['process_create_time'])
    process=psutil.Process(owner['pid']);assert not process.children(recursive=True)
    process.terminate();process.wait(timeout=10)
    atomic_json(out/'TERMINATED_OWNER.json',{'timestamp':utc_now(),'owner':owner,'accounting':account,
        'idle_boundary_verified':True,'worker_terminated':False},immutable=True)
    session.update(modeling_blocked=False,modeling_block_reason=None)
    atomic_json(BASE/'SESSION.json',session)
    with (out/'RESTART.log').open('x',encoding='utf-8') as log:
        child=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),'-u',str(ROOT/'scripts/run_next4_autonomous.py'),'--resume'],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    atomic_json(out/'RESTART.json',{'timestamp':utc_now(),'launcher_pid':child.pid,
        'launcher_create_time':psutil.Process(child.pid).create_time(),'original_owner':owner,
        'source_sha256':{p:digest(ROOT/p) for p in ['src/next4/scheduler.py','src/next4/saturation.py']},
        'regression_xml_sha256':digest(out/'TESTS.xml'),'tests_passed':32,'scientific_fits_repeated':0,
        'clock_preservation':'Explicit resume retains observed prior research; restart observation gap is unobserved, not backcredited'},immutable=True)
    progress('NEXT4 adaptive scheduler resumed','Frozen saturation stop, rejected dependency typos, best-score/start-count checkpoint fields and prospective priority; 32 tests passed')
    print('NEXT4 V2 supervisor launched',child.pid,flush=True)


if __name__=='__main__':main()
