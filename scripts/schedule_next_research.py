"""Dispatch substantive preregistered backlog, preserving the minimum session.

No sleep, busy polling, artificial repeat, or early finalization fallback.
When the finite current backlog is exhausted, return NEEDS_NEW_BACKLOG to the
research agent; that agent must add useful work until the time gate is met.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.research_clock import ResearchClock
from src.next.references import verify_reference_sources
from src.utils.artifacts import write_json,sha256,utc_now


def completed(job):
    if 'queue' in job:
        queue = json.loads((ROOT/job['queue']).read_text(encoding='utf-8'))
        for candidate in queue['candidates']:
            path = ROOT/'artifacts/next/experiments'/candidate['experiment_id']/'RESULTS.json'
            if not path.exists():
                return False
            record = json.loads(path.read_text(encoding='utf-8'))
            if record.get('candidate')!=candidate or record.get('status') not in ('SCREEN_FAIL','FULL_FAIL','BUG','ROBUSTNESS_PENDING','NEW_DEV_CANDIDATE'):
                return False
        return True
    path = ROOT/job['completion_artifact']
    if not path.exists():
        return False
    result = json.loads(path.read_text(encoding='utf-8'))
    return result.get('status')==job.get('completion_status','PASS')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--schedule',default='configs/next_research_schedule.json')
    parser.add_argument('--max-jobs',type=int)
    args = parser.parse_args()
    clock = ResearchClock(ROOT)
    out = clock.directory
    event_file = out/'RESEARCH_EVENTS.jsonl'
    dispatched = 0
    while True:
        verify_reference_sources(ROOT)
        snapshot = clock.snapshot()
        schedule = json.loads((ROOT/args.schedule).read_text(encoding='utf-8'))
        jobs = schedule['jobs']
        if len({job['id'] for job in jobs})!=len(jobs):
            raise RuntimeError('Duplicate research job IDs')
        pending = [job for job in jobs if not completed(job)]
        eligible = [job for job in pending if snapshot['research_elapsed_seconds']>=job.get('not_before_elapsed_seconds',0)]
        if not eligible or (args.max_jobs is not None and dispatched>=args.max_jobs):
            status = 'RESEARCH_MINIMUM_MET_REQUIRES_FINAL_AUDIT' if snapshot['minimum_research_met'] else 'NEEDS_NEW_SUBSTANTIVE_BACKLOG'
            write_json(out/'STATUS.json',{'status':status,**snapshot,'pending_jobs':[job['id'] for job in pending],
                'next_action':'Continue research planning/implementation and add a new bounded job; do not sleep or create a final ZIP.' if not snapshot['minimum_research_met'] else 'Complete required research and explicitly begin the additional final audit.',
                'dispatched_this_run':dispatched})
            print('SCHEDULER',status,'elapsed_hours',round(snapshot['research_elapsed_seconds']/3600,3),flush=True)
            return
        job = eligible[0]
        script = ROOT/job['script']
        if not script.is_file() or script.resolve().parent!=ROOT/'scripts':
            raise RuntimeError('Scheduler only dispatches reviewed local research scripts')
        identity = {'job':job,'script_sha256':sha256(script),'session_sha256':sha256(clock.session_path)}
        manifest = out/(job['id']+'_DISPATCH.json')
        if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8'))!=identity:
            raise RuntimeError('Dispatched job definition changed; register a new revision explicitly')
        if not manifest.exists():
            write_json(manifest,identity)
        event = {'job':job['id'],'started_utc':utc_now(),'monotonic_start':time.monotonic(),
            'research_elapsed_at_start':snapshot['research_elapsed_seconds'],'script_sha256':identity['script_sha256']}
        print('SCHEDULER START',job['id'],flush=True)
        log = out/(job['id']+'.log')
        with log.open('a',encoding='utf-8') as stream:
            process = subprocess.Popen([sys.executable,str(script),*job.get('args',[])],cwd=ROOT,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line.rstrip(),flush=True)
            code = process.wait()
        event.update(completed_utc=utc_now(),elapsed_seconds=time.monotonic()-event['monotonic_start'],
            exit_code=code,completed_artifact_verified=completed(job),log_sha256=sha256(log))
        with event_file.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(event,sort_keys=True)+'\n')
        if code!=0 or not event['completed_artifact_verified']:
            write_json(out/'STATUS.json',{'status':'JOB_REQUIRES_REPAIR','job':job['id'],**clock.snapshot(),'exit_code':code,
                'reason':'Repair or scientifically classify this failure; never convert an error to early session completion.'})
            raise RuntimeError('Research job failed; inspect its retained log')
        dispatched += 1
        write_json(out/'STATUS.json',{'status':'RESEARCH_RUNNING','last_completed_job':job['id'],**clock.snapshot()})


if __name__=='__main__':
    main()
