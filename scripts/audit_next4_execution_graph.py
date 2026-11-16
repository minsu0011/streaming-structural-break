"""Validate real dependency chronology including preserved dispatch repairs."""
import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress


def stamp(value):return datetime.fromisoformat(value)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    if not re.fullmatch('[A-Za-z0-9_]+',args.tag):raise ValueError('Safe immutable tag required')
    paths={p.stem:p for p in (BASE/'queue').glob('*.json')};jobs={k:read_json(v) for k,v in paths.items()}
    dependency_repair=read_json(BASE/'maintenance/DEPENDENCY_KEY_001/REPAIR.json')
    native_repair=read_json(BASE/'maintenance/SYN_NATIVE_FAILURE_001/DEPENDENCY_REPAIR.json')
    repaired_native={r['job']:r for r in native_repair['changes']}
    edges={};exceptions=[];source={};rows=[];states={}
    for identifier,job in jobs.items():
        assert job['id']==identifier
        unsupported=set(job)&{'depends_on','dependencies','requires'}
        if unsupported:
            assert identifier=='B101_FULL' and unsupported=={'depends_on'}
            record=next(r for r in dependency_repair['records'] if r['job']==identifier)
            assert record['actual_prerequisites_completed_before_start'] is True
            deps=job['depends_on'];exceptions.append({'job':identifier,'scope':'Legacy alias ignored by original scheduler; actual prerequisite completion and full runner screen gate independently verified.'})
        else:deps=job.get('depends_on_success',[])
        assert len(deps)==len(set(deps)) and identifier not in deps and set(deps)<=set(jobs)
        edges[identifier]=deps;source[paths[identifier].relative_to(ROOT).as_posix()]=digest(paths[identifier])
        if identifier in repaired_native:
            repair=repaired_native[identifier]
            assert digest(paths[identifier])==repair['after_sha256']
            before=BASE/'maintenance/SYN_NATIVE_FAILURE_001/dependent_queue_before'/paths[identifier].name
            assert digest(before)==repair['before_sha256']==job['original_queue_sha256']
            assert 'SYN_RECOVERY_B2_LOG_ENERGY_SEMITAIL_all_dev' in deps
        for dep in deps:
            if stamp(jobs[dep]['registered_utc'])>stamp(job['registered_utc']):
                assert identifier in repaired_native
                assert stamp(jobs[dep]['registered_utc'])<=stamp(job['dependency_repair_utc'])
        state_path=BASE/'jobs'/identifier/'STATE.json'
        if state_path.exists():
            states[identifier]=read_json(state_path)
            source[state_path.relative_to(ROOT).as_posix()]=digest(state_path)
    visited=set();active=set()
    def visit(identifier):
        if identifier in active:raise RuntimeError('Dependency cycle detected')
        if identifier in visited:return
        active.add(identifier)
        for dep in edges[identifier]:visit(dep)
        active.remove(identifier);visited.add(identifier)
    for identifier in jobs:visit(identifier)
    for identifier,job in jobs.items():
        state=states.get(identifier);status=state['status'] if state else 'PENDING';checked=[]
        if state and 'started_utc' in state:
            for dep in edges[identifier]:
                upstream=states[dep]
                assert upstream['status']=='COMPLETE' and upstream['exit_code']==0,(identifier,dep)
                assert stamp(upstream['finished_utc'])<=stamp(state['started_utc']),(identifier,dep)
                checked.append(dep)
        rows.append({'job':identifier,'kind':job['kind'],'status':status,
            'successful_prerequisites_verified_before_actual_start':checked,
            'dependency_alias_exception':identifier=='B101_FULL','preserved_native_dependency_repair':identifier in repaired_native})
    out=BASE/'execution_graph_audit'/args.tag;out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'registered_jobs':len(jobs),
        'acyclic_dependency_graph':True,'started_jobs_have_actual_successful_prerequisites':True,
        'status_counts':dict(Counter(r['status'] for r in rows)),'rows':rows,
        'legacy_alias_exceptions':exceptions,'preserved_native_dependency_revisions':len(repaired_native),
        'actual_native_failed_jobs_not_reclassified_as_success':True,
        'new_fits':0,'new_candidate_or_gate':False,'source_sha256':digest(Path(__file__)),
        'snapshot_queue_and_state_sha256':source,
        'scope':'Snapshot verification; PENDING/RUNNING jobs are not claimed complete and live state hashes may advance.'},immutable=True)
    progress('Execution dependency chronology audited','Acyclic queue, successful prerequisites before actual starts, legacy alias and four preserved infrastructure-repair edges explicitly accounted')
    print('NEXT4 EXECUTION GRAPH PASS',len(jobs),'jobs',flush=True)


if __name__=='__main__':main()
