"""Preserve failed work and revise only unfinished diagnostic dependencies."""
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next4.scheduler import register_job
from src.next4.synthetic import sources


def main():
    out=BASE/'maintenance/SYN_NATIVE_FAILURE_001'
    original='SYN_B2_LOG_ENERGY_SEMITAIL_all_dev';new='SYN_RECOVERY_B2_LOG_ENERGY_SEMITAIL_all_dev'
    state=read_json(BASE/'jobs'/original/'STATE.json');assert state['status']=='QUARANTINED' and state['exit_code']==3221225477
    probe=read_json(out/'probe/RESULTS.json');assert probe['status']=='DIAGNOSTIC_COMPLETED'
    assert all(r['exit_code']==0 and r['completed_receipt'] for r in probe['runs'])
    a=read_json(out/'probe/normal.json');b=read_json(out/'probe/timed.json')
    assert a['same_array_score_pairs']==b['same_array_score_pairs']
    nomination_id='B2_LOG_ENERGY_SEMITAIL_all_dev';np=BASE/'synthetic_nominations'/(nomination_id+'.json')
    directory=BASE/'synthetic/candidates'/nomination_id
    assert not (directory/'RESULTS.json').exists()
    policy_path=out/'RECOVERY_POLICY.json'
    preserved={p.relative_to(ROOT).as_posix():digest(p) for p in directory.rglob('*') if p.is_file()}
    scientific=sources()
    atomic_json(policy_path,{'created_utc':utc_now(),'original_failed_job':original,'recovery_job':new,
        'nomination':nomination_id,'nomination_sha256':digest(np),
        'preserved_candidate_files_sha256':preserved,'unchanged_scientific_source_sha256':scientific,
        'continuation_source_sha256':digest(ROOT/'scripts/resume_next4_synthetic.py'),
        'worker_instrumentation_sha256':digest(ROOT/'scripts/next4_child_runner.py'),
        'probe_result_sha256':digest(out/'probe/RESULTS.json'),
        'decision':'One same-recipe continuation after a native infrastructure failure; completed outputs are immutable and reused. No partial fit was present or retrained.',
        'native_cause':'Not proven by the short probe. Original stack and event put the crash at the 1500s asynchronous dump in python313.dll. Timed dumps are disabled as an instrumentation-only mitigation.',
        'new_fits':0,'candidate_identity_changed':False,'seed_threshold_or_sample_count_changed':False,
        'automatic_scientific_retry_enabled':False},immutable=True)
    register_job({'id':new,'kind':'validation','priority':105,'family':'B','candidate_id':'B2_LOG_ENERGY_SEMITAIL',
        'depends_on_success':['B501_ALL_DEV_DIAGNOSTIC_REFIT'],
        'argv':['scripts/resume_next4_synthetic.py','--recovery-policy',policy_path.relative_to(ROOT).as_posix()],
        'description':'Bounded continuation of preserved B2 synthetic cohorts after native timed-stack failure, unchanged science',
        'retry_safe':False,'recovery_policy_sha256':digest(policy_path)})
    changes=[];backup=out/'dependent_queue_before';backup.mkdir(exist_ok=False)
    for path in sorted((BASE/'queue').glob('*.json')):
        job=read_json(path)
        if original not in job.get('depends_on_success',[]):continue
        assert not (BASE/'jobs'/job['id']/'STATE.json').exists(),job['id']
        shutil.copyfile(path,backup/path.name);before=digest(path)
        updated={**job,'depends_on_success':[new if dep==original else dep for dep in job['depends_on_success']],
            'dependency_repair_utc':utc_now(),'original_queue_sha256':before,
            'dependency_repair_policy_sha256':digest(policy_path)}
        atomic_json(path,updated)
        changes.append({'job':job['id'],'before_sha256':before,'after_sha256':digest(path)})
    assert len(changes)==4,changes
    atomic_json(out/'DEPENDENCY_REPAIR.json',{'created_utc':utc_now(),'changes':changes,
        'original_failed_job_status_preserved':'QUARANTINED','scores_or_models_changed':False,
        'queued_unstarted_diagnostic_dependencies_only':True},immutable=True)
    progress('Native-failure continuation registered','Same B2 arrays/models/seeds; four unstarted dependent diagnostics now require the explicit recovery job, originals preserved')
    print('NEXT4 SYNTHETIC RECOVERY QUEUED;',len(preserved),'existing files preserved;',len(changes),'diagnostic dependencies revised',flush=True)


if __name__=='__main__':main()
