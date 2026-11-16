"""Preserve invocation failures and clarify the executable fixed cohort count."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next5.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next5.synthetic import cohort_menu


def main():
    menu=read_json(BASE/'synthetic/MENU_LOCK.json');generator=read_json(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json');cohorts=cohort_menu(menu,generator)
    atomic_json(BASE/'maintenance/SYNTHETIC_INVOCATION_RECOVERY.json',{'recorded_utc':utc_now(),'category':'BUG','status':'RESOLVED',
        'failed_invocation':'python -B scripts/freeze_next5_synthetic_execution.py',
        'failure':'System Python has no numba; failed during import before freeze, queue writes or fitted inference.',
        'resolution':'Invoked unchanged script with the existing project .venv/Scripts/python.exe -B; completed without installing packages.',
        'new_fits_or_scores_from_failed_invocation':0,'execution_lock_sha256':digest(BASE/'synthetic/EXECUTION_LOCK.json')},immutable=True)
    atomic_json(BASE/'synthetic/COHORT_COUNT_CLARIFICATION.json',{'recorded_utc':utc_now(),'actual_executable_cohorts':len(cohorts),
        'cohorts':cohorts,'calibration_backgrounds':[r['scenario'] for r in cohorts if r['kind']=='calibration'],
        'clarification':'Queue descriptions and planning prose saying 29 assumed four null backgrounds. The unchanged original DGP contains additional pre-change backgrounds. The executable frozen menu always includes all distinct original pre-change backgrounds.',
        'generators_or_sample_sizes_changed':False,'scores_or_cohorts_dropped':False,
        'menu_sha256':digest(BASE/'synthetic/MENU_LOCK.json'),'execution_sha256':digest(BASE/'synthetic/EXECUTION_LOCK.json')},immutable=True)
    progress('Execution provenance clarification recorded','Correct environment invocation and exact fixed synthetic background count; no changes to fitted recipes, data or execution locks')
    print('NEXT5 EXACT SYNTHETIC COHORTS',len(cohorts),[r['scenario'] for r in cohorts if r['kind']=='calibration'],flush=True)


if __name__=='__main__':main()
