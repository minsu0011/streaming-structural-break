"""Minimum-duration research scheduler with explicit adaptive-backlog handoff.

No sleep or empty work extends the research clock. If new scientific judgment
is needed, the active root researcher supplies a preregistered backlog entry;
the scheduler records NEEDS_ADAPTATION and refuses an audit transition.
"""
from pathlib import Path
import sys
import json
import subprocess
import argparse
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next2.session import clock,checkpoint
from src.utils.artifacts import sha256,write_json,utc_now


def completed(job):
    for name in job['completion_files']:
        path=ROOT/name
        if not path.exists():
            return False
        r=json.loads(path.read_text(encoding='utf-8'))
        if r.get('status') in ('BUG','SCREEN_PASS_FULL_PENDING'):
            return False
    return True


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--backlog',default='artifacts/next2/scheduler/ADAPTIVE_BACKLOG.json')
    args=parser.parse_args()
    out=ROOT/'artifacts/next2/scheduler'
    out.mkdir(parents=True,exist_ok=True)
    while True:
        timing=clock(ROOT)
        if timing['audit_started']:
            raise RuntimeError('Research scheduler cannot run inside final audit')
        queue=json.loads((ROOT/args.backlog).read_text(encoding='utf-8'))
        pending=[j for j in queue['jobs'] if not completed(j)]
        if not pending and timing['minimum_research_met']:
            write_json(out/'RESULTS.json',{'status':'RESEARCH_JOBS_COMPLETE','clock':timing,
                'backlog_sha256':sha256(ROOT/args.backlog),'audit_started':False})
            return
        if not pending:
            write_json(out/'RESULTS.json',{'status':'NEEDS_ADAPTATION','clock':timing,
                'reason':'Finite queue completed before the six-hour minimum. Root researcher must supply the next meaningful hypothesis or diagnostic. Audit/handoff remains blocked.'})
            checkpoint('Adaptive research backlog required',['Write one evidence-based hypothesis','Preregister and execute the next bounded research task'])
            raise SystemExit(75)
        job=pending[0]
        script=(ROOT/job['script']).resolve()
        if not script.is_relative_to((ROOT/'scripts').resolve()) or script.suffix!='.py' or sha256(script)!=job['script_sha256']:
            raise RuntimeError('Research job is not an exact local pinned Python script')
        folder=out/job['id']
        folder.mkdir(exist_ok=True)
        if (folder/'FAILURE.json').exists():
            raise RuntimeError('A failed job requires an explicit reviewed resume entry')
        tick=time.monotonic()
        write_json(folder/'START.json',{'status':'RUNNING','job':job,'clock':timing,'created_utc':utc_now()})
        with (folder/'OUTPUT.log').open('a',encoding='utf-8') as stream:
            process=subprocess.Popen([sys.executable,str(script),*job.get('args',[])],cwd=ROOT,
                stdout=stream,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            while process.poll() is None:
                try:
                    process.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    write_json(folder/'HEARTBEAT.json',{'pid':process.pid,'clock':clock(ROOT),'elapsed_job_seconds':time.monotonic()-tick})
        if process.returncode!=0 or not completed(job):
            write_json(folder/'FAILURE.json',{'status':'FAIL','returncode':process.returncode,
                'elapsed_seconds':time.monotonic()-tick,'completion_contract_met':completed(job)})
            raise RuntimeError('Research job did not complete its declared output contract')
        write_json(folder/'RESULTS.json',{'status':'PASS','elapsed_seconds':time.monotonic()-tick,
            'completion_sha256':{p:sha256(ROOT/p) for p in job['completion_files']}})
        checkpoint('Scheduler job '+job['id'],['Remaining registered work','Adaptive research if queue finishes early'],
            artifacts=[(folder/'RESULTS.json').relative_to(ROOT).as_posix()])


if __name__=='__main__':
    main()
