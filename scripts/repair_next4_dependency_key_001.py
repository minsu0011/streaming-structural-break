"""One-time documented pending dispatch correction; no scientific mutation."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next3.job_dependencies import validate_dependencies


def main():
    out=BASE/'maintenance/DEPENDENCY_KEY_001';assert (out/'BEFORE.json').exists()
    assert not (out/'REPAIR.json').exists();records=[]
    for p in sorted((BASE/'queue').glob('B*.json')):
        j=read_json(p)
        if 'depends_on' not in j:continue
        statepath=BASE/'jobs'/p.stem/'STATE.json'
        if statepath.exists():
            checks=[read_json(BASE/'jobs'/dep/'STATE.json') for dep in j['depends_on']]
            assert all(c['status']=='COMPLETE' and c['finished_utc']<read_json(statepath)['started_utc'] for c in checks)
            records.append({'job':j['id'],'action':'UNCHANGED_STARTED_JOB','actual_prerequisites_completed_before_start':True,
                'prerequisite_receipts':[{k:c[k] for k in ['id','status','finished_utc','exit_code']} for c in checks]})
        else:
            before=digest(p);j['depends_on_success']=j.pop('depends_on');validate_dependencies(j,BASE)
            j['dispatch_metadata_correction_utc']=utc_now();j['prior_queue_sha256']=before
            j['correction_receipt']='artifacts/next4/maintenance/DEPENDENCY_KEY_001/BEFORE.json'
            atomic_json(p,j)
            records.append({'job':j['id'],'action':'PENDING_DISPATCH_KEY_CORRECTED_BEFORE_START',
                'before_sha256':before,'after_sha256':digest(p)})
    atomic_json(out/'REPAIR.json',{'timestamp':utc_now(),'records':records,'scientific_registration_changed':False,
        'features_models_predictions_changed':False,'new_fits_due_to_repair':0,
        'after_source_sha256':digest(ROOT/'scripts/advance_next4_base_stages.py')},immutable=True)
    progress('Stage dependency correction recorded','Pending jobs now require successful worker exit; original metadata preserved; any started jobs verified against earlier successful screens')
    print(records)


if __name__=='__main__':main()
