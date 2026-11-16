"""Link successful repairs to quarantined failures without rewriting failed history."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now


def main():
    cases=[('CI001_COLD_API_DETERMINISM','CI002_B2_READONLY_COLD_VALIDATION',
        BASE/'cold_api_readonly_v2/RESULTS.json',BASE/'maintenance/COLD_API_FAILURE_001/EXCEPTION.json'),
        ('SYN_B2_LOG_ENERGY_SEMITAIL_all_dev','SYN_RECOVERY_B2_LOG_ENERGY_SEMITAIL_all_dev',
         BASE/'synthetic/candidates/B2_LOG_ENERGY_SEMITAIL_all_dev/RESULTS.json',BASE/'maintenance/SYN_NATIVE_FAILURE_001/RECOVERY_POLICY.json')]
    for old,new,result_path,explanation in cases:
        out=BASE/'failure_resolutions'/(old+'.json')
        if out.exists():continue
        new_state_path=BASE/'jobs'/new/'STATE.json'
        if not new_state_path.exists() or not result_path.exists():continue
        new_state=read_json(new_state_path)
        if new_state['status']!='COMPLETE' or new_state['exit_code']!=0:continue
        old_state_path=BASE/'jobs'/old/'STATE.json';old_state=read_json(old_state_path)
        assert old_state['status']=='QUARANTINED' and old_state['exit_code']!=0
        result=read_json(result_path);assert result['status']=='PASS' and result['new_fits']==0
        if old.startswith('SYN_'):
            assert result['all_prior_candidate_bytes_unchanged'] is True
            assert result['scientific_recipe_thresholds_models_seeds_changed'] is False
        else:
            assert result['fresh_B2_processes']==2 and result['all_repeated_predictions_exact'] is True
            assert all(r['cold_import_write_attempts']==0 for r in result['results'])
        evidence=[new_state_path,result_path,explanation,
            ROOT/old_state['attempt_directory']/'RESULT.json',ROOT/old_state['attempt_directory']/'RUN.log']
        atomic_json(out,{'status':'RESOLVED_WITH_PRESERVED_FAILURE','created_utc':utc_now(),
            'original_job':old,'original_exit_code':old_state['exit_code'],'replacement_job':new,
            'original_job_remains_quarantined':True,'quarantined_state_sha256':digest(old_state_path),
            'original_failed_attempt_preserved':True,'new_fits_in_resolution':0,
            'resolution':'Completed explicit repair/continuation; use the linked successful evidence, retain the original failed job as failed history.',
            'source_sha256':digest(Path(__file__)),
            'evidence_sha256':{p.relative_to(ROOT).as_posix():digest(p) for p in evidence}},immutable=True)
        print('NEXT4 PRESERVED FAILURE RESOLVED',old,'via',new,flush=True)


if __name__=='__main__':main()
