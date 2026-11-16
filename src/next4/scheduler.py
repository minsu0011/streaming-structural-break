"""One owned worker, conservative observed research time and immutable attempts."""
import csv
import datetime as dt
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
import uuid
import psutil
from src.next4.artifacts import ROOT,BASE,atomic_json,digest,read_json,utc_now
from src.next3.job_dependencies import validate_dependencies,prerequisites_succeeded
from src.next3.phase_accounting import split_observed_interval
from src.next4.saturation import assess as assess_saturation

HEARTBEAT_SECONDS=20
RESEARCH_SECONDS=21600
AUDIT_SECONDS=2700
IDENTIFIER=re.compile(r'[A-Za-z0-9_-]+\Z')
TERMINAL={'COMPLETE','QUARANTINED','CANCELLED_BY_USER'}


def process_matches(pid,created):
    try:
        process=psutil.Process(pid)
        return process.is_running() and abs(process.create_time()-created)<.1
    except psutil.NoSuchProcess:return False


def append_csv(path,row):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);existed=path.exists()
    with path.open('a',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(row))
        if not existed:writer.writeheader()
        writer.writerow(row);stream.flush();os.fsync(stream.fileno())


def resources():
    mem=psutil.virtual_memory();disk=psutil.disk_usage(str(ROOT))
    return {'timestamp':utc_now(),'RAM_percent':mem.percent,'available_RAM_bytes':mem.available,
        'disk_free_bytes':disk.free,'measuring_process_pid':os.getpid(),
        'measuring_process_RSS_bytes':psutil.Process().memory_info().rss}


def register_job(job):
    if any(k not in job for k in ('id','kind','argv','description')):raise ValueError('Required job fields missing')
    if not isinstance(job['id'],str) or not IDENTIFIER.fullmatch(job['id']):raise ValueError('Safe job ID required')
    if job['kind'] not in ('preflight','research','validation','maintenance','final_audit','delivery'):
        raise ValueError('Explicit task class required')
    if any(key in job for key in ('depends_on','dependencies','requires')):
        raise ValueError('Use depends_on_success for explicit successful-worker prerequisites')
    if 'priority' in job and (type(job['priority']) is not int or not 0<=job['priority']<=1000):
        raise ValueError('Optional priority must be an integer in [0,1000]')
    argv=job['argv']
    if not isinstance(argv,list) or not argv or not all(isinstance(x,str) for x in argv):raise ValueError('Argument list required')
    if argv[0]=='-m':
        if len(argv)<2 or argv[1]!='pytest':raise ValueError('Only pytest module jobs allowed')
    else:
        script=(ROOT/argv[0]).resolve()
        if not script.is_relative_to(ROOT/'scripts') or not script.is_file():raise ValueError('Existing project script required')
    validate_dependencies(job,BASE)
    if job['kind']=='research':
        registration=BASE/'registrations'/(job.get('candidate_id','')+'.json')
        if not registration.is_file():raise ValueError('Research candidate registration required before enqueue')
        record=read_json(registration)
        if record['created_before_score'] is not True:raise ValueError('Prospective candidate registration required')
        job={**job,'registration_sha256':digest(registration)}
    result={**job,'registered_utc':utc_now(),'status':'PENDING'}
    atomic_json(BASE/'queue'/(job['id']+'.json'),result,immutable=True)
    return result


def observe_interval(delta, *, child_progress, recent_activity, work_is_research, end_wall, audit_started_wall=None):
    if not math.isfinite(delta) or delta<0:raise ValueError('Finite nonnegative interval required')
    observed=bool(delta<=2*HEARTBEAT_SECONDS and (child_progress or recent_activity))
    # Administration/delivery signals cannot satisfy the minimum research gate.
    if audit_started_wall is None and not work_is_research:observed=False
    return split_observed_interval(delta,observed=observed,interval_end_wall=end_wall,
        audit_started_wall=audit_started_wall)


class Supervisor:
    def __init__(self,*,resume=False):
        self.child=None;self.job=None;self.log=None;self.hang_requested=False
        self.last_tick=time.monotonic();self.last_resource=self.last_checkpoint=self.last_board_due=0.
        self.last_cpu=0.;self.last_log_size=0;self.last_work=self.last_tick
        session=read_json(BASE/'SESSION.json')
        if session['status'] in ('STOPPED_BY_USER','COMPLETE'):raise RuntimeError('Explicitly stopped/completed NEXT4 cannot auto-resume')
        owner=BASE/'runtime/SUPERVISOR.json'
        if owner.exists():
            previous=read_json(owner)
            if process_matches(previous['pid'],previous['process_create_time']):raise RuntimeError('Live NEXT4 owner exists')
            if not resume:raise RuntimeError('Prior owner requires explicit resume')
        account=BASE/'runtime/ACCOUNTING.json'
        if account.exists():
            if not resume:raise RuntimeError('Preserve existing accounting; resume explicitly')
            old=read_json(account);self.research=old['productive_research_seconds'];self.audit=old['audit_productive_seconds']
            self.unobserved=old['unobserved_seconds']+max(0.,time.time()-old['wall_timestamp'])
        else:
            self.research=self.audit=0.
            self.unobserved=max(0.,time.time()-dt.datetime.fromisoformat(session['started_utc']).timestamp())
        for path in (BASE/'jobs').glob('*/STATE.json'):
            state=read_json(path)
            if state['status']=='RUNNING':
                if process_matches(state['pid'],state['process_create_time']):raise RuntimeError('Live orphan worker requires ownership review')
                atomic_json(path,{**state,'status':'QUARANTINED','reason':'Missing previous worker; preserve partial fits and review before retry','recovered_utc':utc_now()})
        atomic_json(owner,{'pid':os.getpid(),'process_create_time':psutil.Process().create_time(),
            'started_utc':utc_now(),'resume':resume,'source_sha256':digest(Path(__file__))})

    def next_job(self):
        session=read_json(BASE/'SESSION.json')
        if session['status'] in ('STOPPED_BY_USER','COMPLETE'):return None
        saturation_path=BASE/'runtime/SATURATION.json'
        saturation=read_json(saturation_path) if saturation_path.exists() else {}
        for path in sorted((BASE/'queue').glob('*.json'),key=lambda p:(read_json(p).get('priority',100),p.name)):
            job=read_json(path);statepath=BASE/'jobs'/job['id']/'STATE.json';old=read_json(statepath) if statepath.exists() else None
            if old and old['status'] in TERMINAL|{'RUNNING'}:continue
            if old and old['status']=='FAILED':
                allowed=old.get('hang_suspected') is True and old['attempt']<2 and job.get('retry_safe') is True
                if not allowed:
                    atomic_json(statepath,{**old,'status':'QUARANTINED','quarantined_utc':utc_now()});continue
            if not prerequisites_succeeded(job,BASE):continue
            if job['kind']=='research':
                preflight=BASE/'preflight/NEXT4_PREFLIGHT.json'
                if not preflight.is_file() or read_json(preflight)['status']!='PASS' or session.get('modeling_blocked'):continue
                if session.get('audit_started_utc') or self.research>=RESEARCH_SECONDS:continue
                if saturation.get('stop_new_scientific_fits'):continue
                registration=BASE/'registrations'/(job['candidate_id']+'.json')
                if digest(registration)!=job['registration_sha256']:raise RuntimeError('Registered scientific hypothesis changed')
            if job['kind']=='final_audit' and not session.get('audit_started_utc'):continue
            return job,old['attempt']+1 if old else 1
        return None

    def start(self,job,attempt):
        out=BASE/'jobs'/job['id']/('attempt_'+str(attempt));out.mkdir(parents=True,exist_ok=False)
        argv=[x.replace('{attempt_dir}',str(out)) for x in job['argv']]
        command=[sys.executable,'-u',str(ROOT/'scripts/next4_child_runner.py'),'--',*argv]
        self.log=(out/'RUN.log').open('x',encoding='utf-8')
        environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
            NUMEXPR_NUM_THREADS='1',NEXT4_JOB_ID=job['id'],NEXT4_ATTEMPT_DIRECTORY=str(out),PYTHONUNBUFFERED='1')
        self.child=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=self.log,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        self.job={**job,'status':'RUNNING','attempt':attempt,'pid':self.child.pid,
            'process_create_time':psutil.Process(self.child.pid).create_time(),'started_utc':utc_now(),
            'attempt_directory':out.relative_to(ROOT).as_posix(),'command':command}
        atomic_json(BASE/'jobs'/job['id']/'STATE.json',self.job)
        self.last_cpu=0.;self.last_log_size=0;self.last_work=time.monotonic();self.hang_requested=False
        print('NEXT4 START',job['id'],'attempt',attempt,flush=True)

    def inspect_child(self):
        if self.child is None:return False
        code=self.child.poll()
        if code is not None:
            self.log.close();out=ROOT/self.job['attempt_directory']
            result={**self.job,'status':'COMPLETE' if code==0 else 'FAILED','finished_utc':utc_now(),
                'exit_code':code,'hang_suspected':self.hang_requested,'log_sha256':digest(out/'RUN.log')}
            atomic_json(out/'RESULT.json',result,immutable=True);atomic_json(BASE/'jobs'/self.job['id']/'STATE.json',result)
            print('NEXT4',result['status'],self.job['id'],code,flush=True)
            self.child=self.job=self.log=None;return True
        try:parent=psutil.Process(self.child.pid);children=[parent,*parent.children(recursive=True)]
        except psutil.NoSuchProcess:return False
        cpu=0.
        for process in children:
            try:cpu+=sum(process.cpu_times()[:2])
            except psutil.NoSuchProcess:pass
        size=(ROOT/self.job['attempt_directory']/'RUN.log').stat().st_size
        activity=cpu-self.last_cpu>=.03 or size!=self.last_log_size
        now=time.monotonic()
        if activity:self.last_work=now
        self.last_cpu,self.last_log_size=cpu,size
        idle=now-self.last_work
        if idle>=1800 and not self.hang_requested:
            out=ROOT/self.job['attempt_directory'];owned=[]
            for process in children:
                try:owned.append({'pid':process.pid,'create_time':process.create_time()})
                except psutil.NoSuchProcess:pass
            atomic_json(out/'HANG_SUSPECTED.json',{'created_utc':utc_now(),'idle_seconds':idle,'owned':owned,
                'diagnostic_trace_file_excluded_from_progress':True},immutable=True)
            self.hang_requested=True
            for record in reversed(owned):
                if process_matches(record['pid'],record['create_time']):
                    try:psutil.Process(record['pid']).terminate()
                    except psutil.NoSuchProcess:pass
            append_csv(BASE/'runtime/HANG_RECOVERY_LOG.csv',{'timestamp':utc_now(),'job':self.job['id'],
                'attempt':self.job['attempt'],'event':'HANG_SUSPECTED_OWNED_TREE_TERMINATED','idle_seconds':idle})
        return activity or idle<=HEARTBEAT_SECONDS

    def tick(self):
        now=time.monotonic();end_wall=time.time();delta=max(0.,now-self.last_tick)
        job_kind=self.job['kind'] if self.job else None
        child_progress=self.inspect_child();session=read_json(BASE/'SESSION.json')
        activitypath=BASE/'runtime/ACTIVITY.json';activity=read_json(activitypath) if activitypath.exists() else {}
        age=end_wall-dt.datetime.fromisoformat(activity['timestamp']).timestamp() if activity else float('inf')
        recent=0<=age<=90
        research=(job_kind in ('preflight','research','validation')) if job_kind else bool(activity.get('counts_as_research'))
        audit_stamp=dt.datetime.fromisoformat(session['audit_started_utc']).timestamp() if session.get('audit_started_utc') else None
        allocation=observe_interval(delta,child_progress=child_progress,recent_activity=recent,
            work_is_research=research,end_wall=end_wall,audit_started_wall=audit_stamp)
        self.research+=allocation['research_credit_seconds'];self.audit+=allocation['audit_credit_seconds']
        self.unobserved+=allocation['unobserved_seconds'];self.last_tick=now
        wall=end_wall-dt.datetime.fromisoformat(session['started_utc']).timestamp()
        account={'timestamp':utc_now(),'wall_timestamp':end_wall,'monotonic_timestamp':now,'wall_seconds':wall,
            'productive_research_seconds':self.research,'audit_productive_seconds':self.audit,
            'unobserved_seconds':self.unobserved,'last_interval_seconds':delta,'last_allocation':allocation,
            'child_activity':child_progress,'agent_activity_recent':recent,'current_job':self.job['id'] if self.job else None}
        atomic_json(BASE/'runtime/ACCOUNTING.json',account)
        resource=resources()
        states=[read_json(p) for p in (BASE/'jobs').glob('*/STATE.json')]
        candidate_results=[read_json(p) for p in (BASE/'experiments').glob('*/full/RESULTS.json')]
        policy_path=ROOT/'configs/next4/RESEARCH_POLICY_LOCK.json'
        if policy_path.exists():
            saturation=assess_saturation(candidate_results,
                [read_json(p) for p in (BASE/'experiments').glob('*/screen/RESULTS.json')],
                [read_json(p) for p in (BASE/'bootstrap').glob('*/RESULTS.json')],read_json(policy_path))
            saturation_path=BASE/'runtime/SATURATION.json'
            previous=read_json(saturation_path) if saturation_path.exists() else {}
            if {k:v for k,v in previous.items() if k!='updated_utc'}!=saturation:
                atomic_json(saturation_path,{'updated_utc':utc_now(),**saturation})
        if now-self.last_resource>=1800:
            append_csv(BASE/'runtime/RESOURCE_LOG.csv',resource);self.last_resource=now
        if now-self.last_checkpoint>=1800:
            directory=BASE/'checkpoints';number=len(list(directory.glob('NEXT4_CHECKPOINT_*.json')))+1
            scores=[r.get('mean_delta') for r in candidate_results if r.get('mean_delta') is not None]
            budgetpath=ROOT/'configs/next4/CANDIDATE_BUDGET.json'
            atomic_json(directory/f'NEXT4_CHECKPOINT_{number:03d}.json',{'created_utc':utc_now(),**account,
                'current_family':self.job.get('family') if self.job else None,
                'current_hypothesis':self.job.get('candidate_id') if self.job else None,
                'best_delta':max(scores) if scores else None,'full_candidate_count':len(candidate_results),
                'best_score':max((r['mean'] for r in candidate_results),default=None),
                'full_candidates_started':len(list((BASE/'experiments').glob('*/full/START.json'))),
                'registered_candidate_count':len(list((BASE/'registrations').glob('*.json'))),
                'saturation':read_json(BASE/'runtime/SATURATION.json') if (BASE/'runtime/SATURATION.json').exists() else None,
                'screen_count':len(list((BASE/'experiments').glob('*/screen/RESULTS.json'))),
                'failure_count':sum(s['status'] in ('FAILED','QUARANTINED') for s in states),
                'current_bottlenecks':read_json(BASE/'bottlenecks/BOARD.json') if (BASE/'bottlenecks/BOARD.json').exists() else None,
                'candidate_budget_sha256':digest(budgetpath) if budgetpath.exists() else None,
                'seal_status':'All four unopened; hash/guard evidence in preflight',
                'next_task':'Registered successful-dependency queue; adaptive review required if empty',
                'resources':resource},immutable=True);self.last_checkpoint=now
        if now-self.last_board_due>=2700:
            atomic_json(BASE/'bottlenecks/REVIEW_DUE.json',{'timestamp':utc_now(),'productive_research_seconds':self.research,
                'reason':'45-minute bottleneck review; no automatic candidate generation'});self.last_board_due=now
        if self.research>=RESEARCH_SECONDS and not audit_stamp:
            atomic_json(BASE/'runtime/AUDIT_TRANSITION_DUE.json',{'timestamp':utc_now(),'productive_research_seconds':self.research,
                'instruction':'Finish current work, freeze selected scientific roles, then explicitly begin 45-60 minute final audit.'})
        if self.child is None and resource['RAM_percent']<85 and resource['disk_free_bytes']>10*1024**3:
            candidate=self.next_job()
            if candidate:self.start(*candidate)
        return session['status'] in ('STOPPED_BY_USER','COMPLETE') and self.child is None

    def run(self):
        print('NEXT4 SUPERVISOR ACTIVE',os.getpid(),flush=True)
        while True:
            try:
                if self.tick():return
            except Exception as error:
                atomic_json(BASE/'runtime/errors'/(uuid.uuid4().hex+'.json'),{'created_utc':utc_now(),
                    'error':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()},immutable=True)
                session=read_json(BASE/'SESSION.json');session.update(modeling_blocked=True,modeling_block_reason='Supervisor error requires repair review')
                atomic_json(BASE/'SESSION.json',session)
                print('NEXT4 SUPERVISOR ERROR',type(error).__name__,str(error),flush=True)
            time.sleep(HEARTBEAT_SECONDS)  # Monitoring cadence is never productive evidence by itself.
