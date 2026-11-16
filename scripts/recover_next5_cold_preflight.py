"""Preserve the missing -B failure and register a read-only cold-check recovery."""
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next5.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next5.scheduler import register_job


def main():
    if len(sys.argv)>1:
        role=sys.argv[1]
        jobs={
            'BL80':['scripts/verify_next4_cold_api.py','--worker','BL80','--output','artifacts/next5/preflight/BL80_COLD.json'],
            'B2':['scripts/verify_next4_b2_readonly.py','--worker-output','artifacts/next5/preflight/B2_COLD.json'],
            'D2':['scripts/verify_next4_candidate_api.py','--model','D2_DISCRETE_SCALE_MIXTURE','--cold-worker-output','artifacts/next5/preflight/D2_COLD.json']}
        subprocess.run([sys.executable,'-B','-u',*jobs[role]],cwd=ROOT,check=True)
        return
    failed=BASE/'jobs/PF003_BL80_COLD/STATE.json'
    assert read_json(failed)['status']=='QUARANTINED'
    preserved={}
    for name in ('PF004_B2_COLD','PF005_D2_COLD','PF006_FINALIZE'):
        path=BASE/'jobs'/name/'STATE.json'
        assert not path.exists()
        queue=BASE/'queue'/(name+'.json');preserved[name]=digest(queue)
        atomic_json(path,{**read_json(queue),'status':'QUARANTINED','reason':'Unstarted dependency chain superseded by explicitly recorded -B recovery','resolved_by':'PF016_FINALIZE_RECOVERY','created_utc':utc_now()},immutable=True)
    atomic_json(BASE/'preflight/COLD_FAILURE_RECOVERY.json',{
        'created_utc':utc_now(),'failed_job':'PF003_BL80_COLD','failed_state_sha256':digest(failed),
        'reason':'Existing cold worker requires sys.dont_write_bytecode; scheduler child did not pass -B.',
        'failure_before_model_inference':True,'new_fits':0,'shadow_rows':0,'seal_rows':0,
        'recovery':'New wrapper starts original unchanged worker with python -B; original failure and queue bytes preserved.',
        'superseded_queue_sha256':preserved,'resolved_by':'PF016_FINALIZE_RECOVERY'},immutable=True)
    dependency='PF001_GUARD_SCORER_TESTS'
    for number,role in enumerate(('BL80','B2','D2'),13):
        name=f'PF0{number}_{role}_COLD_RECOVERY'
        register_job({'id':name,'kind':'preflight','priority':number,'depends_on_success':[dependency],
            'argv':['scripts/recover_next5_cold_preflight.py',role],
            'description':'Preserved failure recovery: synthetic cold '+role+' with explicit python -B','retry_safe':False})
        dependency=name
    register_job({'id':'PF016_FINALIZE_RECOVERY','kind':'preflight','priority':16,
        'depends_on_success':['PF002_RESEARCH_RAW',dependency],
        'argv':['scripts/finish_next5_preflight.py'],'description':'Finalize actual guard/scorer/three cold checks with failure preserved','retry_safe':False})
    progress('Preflight cold process invocation corrected','Original missing -B failure preserved; source and legacy models unchanged',phase='PREFLIGHT')


if __name__=='__main__':main()
