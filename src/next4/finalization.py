"""Explicit phase transition and handoff checks; no retroactive clock credit."""
from datetime import datetime
import math
import os
import time
from src.next4.artifacts import ROOT,BASE,read_json,atomic_json,digest,utc_now

RESEARCH_MIN_SECONDS=21600
AUDIT_MIN_SECONDS=2700


def require_timing(account,session,*,now_wall):
    if not math.isfinite(now_wall):raise ValueError('Finite current clock required')
    for field in ('productive_research_seconds','audit_productive_seconds','unobserved_seconds','wall_seconds','wall_timestamp'):
        value=account[field]
        if not isinstance(value,(float,int)) or not math.isfinite(value) or value<0:
            raise ValueError('Invalid observed accounting value: '+field)
    age=now_wall-account['wall_timestamp']
    if not 0<=age<=60:raise RuntimeError('Fresh actual supervisor accounting required')
    if account['productive_research_seconds']<RESEARCH_MIN_SECONDS:
        raise RuntimeError('At least 21600 observed productive research seconds required')
    if not session.get('audit_started_utc'):raise RuntimeError('Separate actual audit start required')
    audit_start=datetime.fromisoformat(session['audit_started_utc']).timestamp()
    audit_wall=account['wall_timestamp']-audit_start
    if audit_wall<AUDIT_MIN_SECONDS or account['audit_productive_seconds']<AUDIT_MIN_SECONDS:
        raise RuntimeError('At least 2700 separate observed final-audit seconds required')
    if account['audit_productive_seconds']>audit_wall+1:
        raise RuntimeError('Audit credit cannot precede its actual start')
    if sum(account[k] for k in ('productive_research_seconds','audit_productive_seconds','unobserved_seconds'))>account['wall_seconds']+2:
        raise RuntimeError('Observed plus unobserved accounting exceeds elapsed wall time')
    return {'productive_research_seconds':account['productive_research_seconds'],
        'audit_productive_seconds':account['audit_productive_seconds'],'audit_wall_seconds':audit_wall,
        'wall_seconds':account['wall_seconds'],'unobserved_seconds':account['unobserved_seconds'],
        'accounted_utc':account['timestamp'],'audit_started_utc':session['audit_started_utc'],
        'research_minimum_pass':True,'separate_audit_minimum_pass':True,
        'audit_within_60_minutes_at_check':audit_wall<=3600,'no_retroactive_time_credit':True}


def require_handoff_time():
    return require_timing(read_json(BASE/'runtime/ACCOUNTING.json'),read_json(BASE/'SESSION.json'),now_wall=time.time())


def begin_audit():
    session_path=BASE/'SESSION.json';session=read_json(session_path)
    if session.get('audit_started_utc'):raise RuntimeError('Never replace or backdate an audit start')
    if session['status']!='ACTIVE_RESEARCH':raise RuntimeError('Only an active research session can enter final audit')
    account=read_json(BASE/'runtime/ACCOUNTING.json')
    if not 0<=time.time()-account['wall_timestamp']<=60:raise RuntimeError('Fresh actual accounting required')
    if account['productive_research_seconds']<RESEARCH_MIN_SECONDS:raise RuntimeError('Research minimum is incomplete')
    selection_path=ROOT/'DEV_CANDIDATES_NEXT4.json';selection=read_json(selection_path)
    if selection['status']!='FROZEN_NEXT4_DEV_SELECTION':raise RuntimeError('Reviewed final scientific roles must be frozen before audit')
    pending=[];resolved=[]
    for path in sorted((BASE/'queue').glob('*.json')):
        job=read_json(path)
        if job['kind'] in ('final_audit','delivery'):continue
        state_path=BASE/'jobs'/job['id']/'STATE.json'
        state=read_json(state_path) if state_path.exists() else {'status':'PENDING'}
        if state['status']=='COMPLETE' and state.get('exit_code')==0:continue
        resolution_path=BASE/'failure_resolutions'/(job['id']+'.json')
        if state['status']=='QUARANTINED' and resolution_path.exists():
            resolution=read_json(resolution_path)
            if resolution['status']=='RESOLVED_WITH_PRESERVED_FAILURE' and digest(state_path)==resolution['quarantined_state_sha256']:
                for name,sha in resolution['evidence_sha256'].items():
                    if digest(ROOT/name)!=sha:raise RuntimeError('Failure-resolution evidence changed')
                resolved.append({'job':job['id'],'resolution_sha256':digest(resolution_path)});continue
        pending.append({'job':job['id'],'status':state['status']})
    if pending:raise RuntimeError('Required work remains before audit: '+str(pending))
    out=BASE/'phase_transitions/RESEARCH_TO_FINAL_AUDIT';out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'SESSION_BEFORE.json',session,immutable=True)
    atomic_json(out/'ACCOUNTING_BEFORE.json',account,immutable=True)
    stamp=utc_now();updated={**session,'status':'ACTIVE_FINAL_AUDIT','audit_started_utc':stamp,
        'audit_research_seconds_at_transition':account['productive_research_seconds'],
        'final_selection_sha256':digest(selection_path),'finalization_source_sha256':digest(ROOT/'src/next4/finalization.py')}
    atomic_json(session_path,updated)
    atomic_json(out/'TRANSITION.json',{'status':'PASS','created_utc':stamp,'pid':os.getpid(),
        'research_productive_seconds_observed_before_transition':account['productive_research_seconds'],
        'audit_started_utc':stamp,'audit_time_backdated':False,'pending_required_jobs':0,
        'preserved_resolved_failures':resolved,'selection_sha256':digest(selection_path),
        'source_sha256':digest(ROOT/'src/next4/finalization.py')},immutable=True)
    return updated
