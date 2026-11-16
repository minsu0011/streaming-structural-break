"""Single-worker supervisor, conservative activity accounting and crash receipts."""
from pathlib import Path
import csv
import datetime
import faulthandler
import json
import os
import subprocess
import sys
import time
import traceback
import uuid
import psutil
from src.next3.artifacts import ROOT, BASE, atomic_json, read_json, digest, utc_now
from src.next3.job_dependencies import validate_dependencies, prerequisites_succeeded
from src.next3.phase_accounting import split_observed_interval

HEARTBEAT_SECONDS = 45
RESEARCH_SECONDS = 40200
AUDIT_SECONDS = 3000


def append_csv(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open('a', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(record))
        if not exists:
            writer.writeheader()
        writer.writerow(record)
        stream.flush()
        os.fsync(stream.fileno())


def resource_snapshot():
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage(str(ROOT))
    return {'timestamp': utc_now(), 'RAM_percent': memory.percent,
        'available_RAM_bytes': memory.available, 'disk_free_bytes': disk.free,
        'supervisor_RSS_bytes': psutil.Process().memory_info().rss,
        'child_process_count': len(psutil.Process().children(recursive=True))}


def register_job(job):
    required = ('id', 'kind', 'argv', 'description')
    if any(key not in job for key in required):
        raise ValueError('Job fields missing')
    name = job['id']
    if not name or any(c not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-' for c in name):
        raise ValueError('Unsafe job identifier')
    if job['kind'] == 'research':
        if not (BASE/'preflight/NEXT3_PREFLIGHT.json').exists():
            raise RuntimeError('Modeling blocked before preflight')
        hypothesis = ROOT/job['hypothesis']
        if not hypothesis.resolve().is_relative_to(BASE.resolve()):
            raise ValueError('Hypothesis must be registered in NEXT3')
        job['hypothesis_sha256'] = digest(hypothesis)
    argv = job['argv']
    if not isinstance(argv, list) or not argv or not all(isinstance(arg, str) for arg in argv):
        raise ValueError('Explicit Python argument list required; no shell strings')
    if argv[0] == '-m':
        if len(argv) < 2 or argv[1] != 'pytest':
            raise ValueError('Only pytest module execution is accepted')
    elif not (ROOT/argv[0]).resolve().is_relative_to((ROOT/'scripts').resolve()):
        raise ValueError('Child must be a project script')
    validate_dependencies(job, BASE)
    job = {**job, 'registered_utc': utc_now(), 'status': 'PENDING'}
    atomic_json(BASE/'queue'/f'{name}.json', job, immutable=True)
    return job


def activity_age():
    path = BASE/'runtime/ACTIVITY.json'
    if not path.exists():
        return float('inf')
    try:
        stamp = datetime.datetime.fromisoformat(read_json(path)['timestamp']).timestamp()
        return max(0., time.time()-stamp)
    except (ValueError, KeyError):
        return float('inf')


class Supervisor:
    def __init__(self, *, resume=False):
        self.session = read_json(BASE/'SESSION.json')
        self.child = None
        self.job = None
        self.child_log = None
        self.last_tick = time.monotonic()
        self.last_resource = 0.
        self.last_checkpoint = 0.
        self.last_reprioritization = 0.
        self.last_progress = self.last_tick
        self.last_cpu = 0.
        self.last_log_size = 0
        self.last_cpu_progress = self.last_tick
        self.running_path = BASE/'runtime/SUPERVISOR.json'
        if self.running_path.exists():
            prior = read_json(self.running_path)
            if psutil.pid_exists(prior['pid']):
                proc = psutil.Process(prior['pid'])
                if abs(proc.create_time()-prior['process_create_time']) < .1:
                    raise RuntimeError('A live supervisor already owns this run')
            if not resume:
                raise RuntimeError('Previous supervisor exists; use explicit resume')
        previous = BASE/'runtime/ACCOUNTING.json'
        if previous.exists():
            if not resume:
                raise RuntimeError('Existing accounting requires resume')
            accounting = read_json(previous)
            self.productive = accounting['productive_seconds']
            self.audit_productive = accounting.get('audit_productive_seconds', 0.)
            self.unobserved = accounting['unobserved_seconds'] + max(0., time.time()-accounting['wall_timestamp'])
            self._recover_jobs()
        else:
            # This initial interval is explicitly reviewed active setup work;
            # subsequent time needs observed activity, never a timer alone.
            initial = self.last_tick-self.session['monotonic_start']
            if initial < 0 or initial > 1800 or activity_age() > 180:
                raise RuntimeError('Initial setup interval cannot be safely credited')
            self.productive = initial
            self.audit_productive = 0.
            self.unobserved = 0.
            atomic_json(BASE/'runtime/INITIAL_PRODUCTIVE_INTERVAL.json', {
                'start_utc': self.session['started_utc'], 'end_utc': utc_now(), 'seconds': initial,
                'work': 'Frozen restoration, guard/scorer tests, public replay, implementation of maintenance supervisor',
                'basis': 'Continuous active agent implementation/tool session, supported by initialization/test/replay files'}, immutable=True)
        atomic_json(self.running_path, {'pid': os.getpid(), 'process_create_time': psutil.Process().create_time(),
            'started_utc': utc_now(), 'resume': resume, 'source_sha256': digest(Path(__file__))})

    def _recover_jobs(self):
        for path in sorted((BASE/'jobs').glob('*/STATE.json')):
            state = read_json(path)
            if state['status'] != 'RUNNING':
                continue
            pid = state.get('pid')
            if pid and psutil.pid_exists(pid):
                process = psutil.Process(pid)
                if abs(process.create_time()-state.get('process_create_time', -1)) < .1:
                    raise RuntimeError('Owned child still running: inspect/rejoin before restart')
            state.update(status='FAILED', reason='Supervisor/child interrupted; previous attempt preserved', recovered_utc=utc_now())
            atomic_json(path, state)
            append_csv(BASE/'runtime/HANG_RECOVERY_LOG.csv', {'timestamp': utc_now(), 'job': state['id'],
                'attempt': state['attempt'], 'event': 'CRASH_RECOVERED', 'reason': state['reason']})

    def states(self):
        return [read_json(p) for p in sorted((BASE/'jobs').glob('*/STATE.json'))]

    def next_job(self):
        session = read_json(BASE/'SESSION.json')
        elapsed = time.time()-datetime.datetime.fromisoformat(session['started_utc']).timestamp()
        for path in sorted((BASE/'queue').glob('*.json')):
            job = read_json(path)
            state_path = BASE/'jobs'/job['id']/'STATE.json'
            old = read_json(state_path) if state_path.exists() else None
            if old and old['status'] in ('COMPLETE', 'QUARANTINED', 'RUNNING'):
                continue
            if old and old['status'] == 'FAILED' and (old['attempt'] >= 2 or not job.get('retry_safe', False)):
                atomic_json(state_path, {**old, 'status': 'QUARANTINED', 'quarantined_utc': utc_now()})
                continue
            if elapsed >= RESEARCH_SECONDS and job['kind'] == 'research':
                continue
            if job['kind'] == 'research':
                if session.get('modeling_blocked', False):
                    continue
                assert read_json(BASE/'preflight/NEXT3_PREFLIGHT.json')['status'] == 'PASS'
                assert digest(ROOT/job['hypothesis']) == job['hypothesis_sha256']
            if any(not (ROOT/name).exists() for name in job.get('requires', [])):
                continue
            if not prerequisites_succeeded(job, BASE):
                continue
            return job, (old['attempt'] + 1 if old else 1)
        return None

    def start(self, job, attempt):
        directory = BASE/'jobs'/job['id']/f'attempt_{attempt}'
        directory.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, '-u', str(ROOT/'scripts/next3_child_runner.py'), '--'] + [
            arg.replace('{attempt_dir}', str(directory)) for arg in job['argv']]
        self.child_log = (directory/'RUN.log').open('w', encoding='utf-8')
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
            NEXT3_JOB_ID=job['id'], NEXT3_ATTEMPT_DIRECTORY=str(directory), PYTHONUNBUFFERED='1')
        self.child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=self.child_log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.job = {**job, 'status': 'RUNNING', 'attempt': attempt, 'pid': self.child.pid,
            'process_create_time': psutil.Process(self.child.pid).create_time(), 'started_utc': utc_now(),
            'attempt_directory': directory.relative_to(ROOT).as_posix(), 'command': command}
        atomic_json(BASE/'jobs'/job['id']/'STATE.json', self.job)
        self.last_progress = self.last_cpu_progress = time.monotonic()
        self.last_cpu = 0.
        self.last_log_size = 0
        print('NEXT3 START', job['id'], 'attempt', attempt, flush=True)

    def inspect_child(self):
        if self.child is None:
            return False
        code = self.child.poll()
        if code is not None:
            self.child_log.close()
            result = {**self.job, 'status': 'COMPLETE' if code == 0 else 'FAILED',
                'finished_utc': utc_now(), 'exit_code': code,
                'log_sha256': digest(ROOT/self.job['attempt_directory']/'RUN.log')}
            atomic_json(ROOT/self.job['attempt_directory']/'RESULT.json', result, immutable=True)
            atomic_json(BASE/'jobs'/self.job['id']/'STATE.json', result)
            if code and self.job.get('critical_preflight', False):
                session = read_json(BASE/'SESSION.json')
                session.update(modeling_blocked=True, modeling_block_reason=self.job['id'])
                atomic_json(BASE/'SESSION.json', session)
            print('NEXT3', result['status'], self.job['id'], 'exit', code, flush=True)
            self.child = self.job = self.child_log = None
            return True
        try:
            process = psutil.Process(self.child.pid)
            processes = [process] + process.children(recursive=True)
        except psutil.NoSuchProcess:
            # Exit can occur between poll and process inspection. The next
            # poll finalizes the child; this is not a modeling catastrophe.
            return False
        cpu = 0.
        for child_process in processes:
            try:
                cpu += sum(child_process.cpu_times()[:2])
            except psutil.NoSuchProcess:
                continue
        size = (ROOT/self.job['attempt_directory']/'RUN.log').stat().st_size
        now = time.monotonic()
        changed = size != self.last_log_size
        busy = cpu-self.last_cpu >= .05
        if changed:
            self.last_progress = now
        if busy:
            self.last_cpu_progress = now
        self.last_cpu, self.last_log_size = cpu, size
        idle = now-max(self.last_progress, self.last_cpu_progress)
        if idle >= 1800:
            diagnostic = {'timestamp': utc_now(), 'event': 'HANG_SUSPECTED', 'job': self.job['id'],
                'attempt': self.job['attempt'], 'no_progress_or_CPU_seconds': idle,
                'owned_processes': [{'pid': p.pid, 'command': p.cmdline()} for p in processes],
                'log_tail': (ROOT/self.job['attempt_directory']/'RUN.log').read_text(encoding='utf-8', errors='replace')[-16000:]}
            atomic_json(ROOT/self.job['attempt_directory']/'HANG_SUSPECTED.json', diagnostic, immutable=True)
            # Only the owned experiment and its descendants; no unrelated process.
            for owned in reversed(processes):
                try:
                    owned.terminate()
                except psutil.NoSuchProcess:
                    pass
            append_csv(BASE/'runtime/HANG_RECOVERY_LOG.csv', {'timestamp': utc_now(), 'job': self.job['id'],
                'attempt': self.job['attempt'], 'event': 'HANG_SUSPECTED_CHILD_TERMINATED', 'reason': '1800 seconds without output or CPU progress'})
        return changed or busy or now-self.last_progress <= HEARTBEAT_SECONDS

    def checkpoint(self, heartbeat):
        directory = BASE/'checkpoints'
        directory.mkdir(parents=True, exist_ok=True)
        number = len(list(directory.glob('NEXT3_CHECKPOINT_*.json'))) + 1
        report = {**heartbeat, 'checkpoint_number': number,
            'seal_status': {'SEAL_A': False, 'SEAL_B': False, 'SEAL_C': False, 'SEAL_D': False},
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'accounting_sha256': digest(BASE/'runtime/ACCOUNTING.json'),
            'jobs': [{k: s.get(k) for k in ('id', 'status', 'attempt', 'finished_utc')} for s in self.states()]}
        atomic_json(directory/f'NEXT3_CHECKPOINT_{number:03d}.json', report, immutable=True)

    def tick(self):
        now = time.monotonic()
        interval_end_wall = time.time()
        delta = max(0., now-self.last_tick)
        child_active = self.inspect_child()
        session = read_json(BASE/'SESSION.json')
        phase = 'FINAL_AUDIT' if session.get('audit_started_utc') else session['status']
        recent_activity = activity_age() <= 180
        # Suspension/debugger gaps cannot inflate productive time.
        observed = delta <= HEARTBEAT_SECONDS*2 and (child_active or recent_activity)
        audit_started_wall = datetime.datetime.fromisoformat(session['audit_started_utc']).timestamp() if session.get('audit_started_utc') else None
        allocation = split_observed_interval(delta, observed=observed,
            interval_end_wall=interval_end_wall, audit_started_wall=audit_started_wall)
        credit = allocation['observed_credit_seconds']
        self.productive += allocation['research_credit_seconds']
        self.audit_productive += allocation['audit_credit_seconds']
        self.unobserved += allocation['unobserved_seconds']
        self.last_tick = now
        wall = time.time()-datetime.datetime.fromisoformat(session['started_utc']).timestamp()
        accounting = {'timestamp': utc_now(), 'wall_timestamp': time.time(), 'monotonic_timestamp': now,
            'wall_seconds': wall, 'productive_seconds': self.productive, 'audit_productive_seconds': self.audit_productive,
            'unobserved_seconds': self.unobserved, 'last_interval_seconds': delta, 'last_interval_credited': credit,
            'last_interval_research_credit_seconds': allocation['research_credit_seconds'],
            'last_interval_audit_credit_seconds': allocation['audit_credit_seconds'],
            'interval_end_wall_for_phase_allocation': interval_end_wall,
            'child_activity': child_active, 'recent_agent_or_worker_activity': recent_activity}
        atomic_json(BASE/'runtime/ACCOUNTING.json', accounting)
        resources = resource_snapshot()
        states = self.states()
        activity = read_json(BASE/'runtime/ACTIVITY.json')
        heartbeat = {**accounting, 'pid': os.getpid(), 'current_phase': phase,
            'current_experiment': self.job['id'] if self.job else None,
            'last_progress': activity, 'resources': resources,
            'completed_count': sum(s['status'] == 'COMPLETE' for s in states),
            'failed_count': sum(s['status'] in ('FAILED', 'QUARANTINED') for s in states),
            'current_best': session.get('current_best', 'FROZEN_PERFORMANCE'),
            'next_task': 'Run eligible registered queue; otherwise agent reviews evidence and registers the next task',
            'research_launch_closed': wall >= RESEARCH_SECONDS, 'concurrency': 1}
        atomic_json(BASE/'runtime/HEARTBEAT.json', heartbeat)
        if now-self.last_resource >= 1800:
            resources['next3_temp_bytes'] = sum(p.stat().st_size for p in BASE.rglob('*.tmp') if p.is_file())
            append_csv(BASE/'runtime/RESOURCE_LOG.csv', resources)
            self.last_resource = now
        if now-self.last_checkpoint >= 1800:
            self.checkpoint(heartbeat)
            self.last_checkpoint = now
        if now-self.last_reprioritization >= 2700:
            atomic_json(BASE/'bottlenecks/REPRIORITIZATION_DUE.json', {
                'timestamp': utc_now(), 'wall_seconds': wall, 'reason': '45 minute review; reassess evidence before additional hypothesis families'})
            self.last_reprioritization = now
        if wall >= RESEARCH_SECONDS and not session.get('audit_started_utc'):
            atomic_json(BASE/'runtime/AUDIT_TRANSITION_DUE.json', {
                'timestamp': utc_now(), 'wall_seconds': wall, 'active_child': self.job['id'] if self.job else None,
                'instruction': 'Finish current child safely; no new large scientific experiments. Agent begins separate final audit.'})
        if self.child is None and session.get('status') != 'COMPLETE' and resources['RAM_percent'] < 85 and resources['disk_free_bytes'] > 10*1024**3:
            candidate = self.next_job()
            if candidate:
                self.start(*candidate)
        return session.get('status') == 'COMPLETE' and self.child is None

    def run(self):
        print('NEXT3 SUPERVISOR ACTIVE', os.getpid(), flush=True)
        while True:
            try:
                if self.tick():
                    print('NEXT3 SUPERVISOR COMPLETE', flush=True)
                    return
            except Exception as error:
                receipt = {'timestamp': utc_now(), 'event': 'SUPERVISOR_ERROR',
                    'error_type': type(error).__name__, 'message': str(error),
                    'traceback': traceback.format_exc(),
                    'child_pid': self.child.pid if self.child else None,
                    'modeling_action': 'New scientific launches blocked until reviewed; existing owned child preserved'}
                atomic_json(BASE/'runtime/errors'/f'{uuid.uuid4().hex}.json', receipt, immutable=True)
                session = read_json(BASE/'SESSION.json')
                session.update(modeling_blocked=True, modeling_block_reason='Supervisor error requires documented repair')
                atomic_json(BASE/'SESSION.json', session)
                print('NEXT3 SUPERVISOR ERROR RECORDED', type(error).__name__, str(error), flush=True)
            # Monitoring cadence, not credited research by itself.
            time.sleep(HEARTBEAT_SECONDS)
