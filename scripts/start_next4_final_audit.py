"""Enter the actual separate audit phase and enqueue its finite required work."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next4.finalization import begin_audit
from src.next4.scheduler import register_job


def main():
    selection=read_json(ROOT/'DEV_CANDIDATES_NEXT4.json');role=selection['roles'].get('NEXT4_ROBUSTNESS')
    full=sorted(p.parent.name for p in (BASE/'bootstrap').glob('*/RESULTS.json') if p.parent.name!='frozen_BL80')
    assert len(full)==14
    jobs=[('FA001_FROZEN_BOUNDARY',['scripts/audit_next4_frozen_boundary.py','--tag','final_v1']),
        ('FA002_PRIMARY_ORIGINAL_DRAWS',['scripts/audit_next4_primary_statistics.py','--tag','final_v1','--candidates',*full]),
        ('FA002B_MECHANISM_ORIGINAL_DRAWS',['scripts/audit_next4_mechanism_statistics.py','--tag','final_v1']),
        ('FA002C_NOMINATION_CHRONOLOGY',['scripts/audit_next4_nomination_chronology.py','--tag','final_v1']),
        ('FA003_FROZEN_PUBLIC_API',['scripts/verify_next4_candidate_api.py','--model','BL80','--tag','final_v1','--audit']),
        ('FA004_B2_PUBLIC_API',['scripts/verify_next4_candidate_api.py','--model','B2','--tag','final_v1','--audit'])]
    if role:
        cid=role['experiment_id']
        jobs.extend([('FA005_ROBUST_PUBLIC_API',['scripts/verify_next4_candidate_api.py','--model',cid,'--tag','final_v1','--audit']),
            ('FA006_FRESH_RAW_REPRODUCTION',['scripts/materialize_next4_robust_finalist.py','--candidate',cid,'--fresh-raw']),
            ('FA007_SELECTED_DIRECT_TIMING',['scripts/profile_next4_robust_finalist.py','--candidate',cid,'--audit'])])
    else:jobs.append(('FA007_FROZEN_DIRECT_TIMING',['scripts/profile_next4_frozen_audit.py']))
    jobs.extend([('FA008_COMPLETE_TEST_SUITE',['scripts/run_next4_final_tests.py','--tag','final_v1']),
        ('FA009_COPIED_PAYLOAD',['scripts/verify_next4_portable_payload.py','--tag','final_v1'])])
    # Validate the files before the irreversible actual phase timestamp is written.
    for identifier,argv in jobs:assert (ROOT/argv[0]).is_file() and not (BASE/'queue'/(identifier+'.json')).exists()
    session=begin_audit()
    atomic_json(BASE/'final_audit/PLAN.json',{'created_utc':utc_now(),'audit_started_utc':session['audit_started_utc'],
        'selection_sha256':digest(ROOT/'DEV_CANDIDATES_NEXT4.json'),'jobs':[j[0] for j in jobs],
        'primary_statistics_scope':'All 14 original full candidates; exactly original 2000 multiplicities, no added draws.',
        'minimum_observed_audit_seconds':2700,'target_maximum_wall_seconds':3600,
        'source_sha256':digest(Path(__file__))},immutable=True)
    previous=[]
    for index,(identifier,argv) in enumerate(jobs):
        register_job({'id':identifier,'kind':'final_audit','priority':200+index,'depends_on_success':previous,
            'argv':argv,'description':'Separate final audit '+identifier,'retry_safe':False})
        previous=[identifier]
    progress('Separate final audit started','Actual research time gate satisfied; original scores, selected APIs, raw/guard/source, reproduction if selected, full tests and moved payload',phase='FINAL_AUDIT',counts_as_research=False)
    print('NEXT4 ACTUAL FINAL AUDIT STARTED',session['audit_started_utc'],flush=True)


if __name__=='__main__':main()
