"""Close NEXT4 only after verified archives and all required jobs are complete."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now
from src.next4.finalization import require_handoff_time
from src.next.guard import SealGuard


def main():
    session=read_json(BASE/'SESSION.json');assert session['status']=='ACTIVE_FINAL_AUDIT'
    timing=require_handoff_time();audit=read_json(ROOT/'FINAL_AUDIT_NEXT4.json')
    assert audit['status']=='PASS' and all(v is True for v in audit['checks'].values())
    for name,sha in audit['evidence_sha256'].items():assert digest(ROOT/name)==sha
    pointer=read_json(BASE/'handoff/DESKTOP_DELIVERY_POINTER.json');path=(ROOT/pointer['receipt']).resolve()
    assert path.is_relative_to((BASE/'handoff').resolve()) and digest(path)==pointer['sha256']
    delivery=read_json(path);assert delivery['status']=='PASS'
    for key in ('full_project_archive','twenty_file_archive'):
        record=delivery[key];archive=Path(record['path'])
        assert archive.is_file() and archive.stat().st_size==record['bytes'] and digest(archive)==record['sha256']
        assert record['CRC_verified'] and record['all_member_SHA256_verified']
    folder=Path(delivery['twenty_file_folder']);files=list(folder.iterdir())
    assert len(files)==20 and all(p.is_file() for p in files)
    for name,record in delivery['twenty_files'].items():assert digest(folder/name)==record['sha256']
    completed=0;resolved=[]
    for path in (BASE/'queue').glob('*.json'):
        job=read_json(path);statepath=BASE/'jobs'/job['id']/'STATE.json';state=read_json(statepath)
        if state['status']=='COMPLETE' and state['exit_code']==0:completed+=1;continue
        resolution=read_json(BASE/'failure_resolutions'/(job['id']+'.json'))
        assert state['status']=='QUARANTINED' and resolution['status']=='RESOLVED_WITH_PRESERVED_FAILURE'
        assert resolution['quarantined_state_sha256']==digest(statepath)
        for name,sha in resolution['evidence_sha256'].items():assert digest(ROOT/name)==sha
        resolved.append(job['id'])
    SealGuard(ROOT).check_state();old=read_json(ROOT/'artifacts/next3/SESSION.json')
    assert old['status']=='STOPPED_BY_USER' and old['automatic_resume_authorized'] is False
    stamp=utc_now();out=BASE/'completion';out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'SESSION_BEFORE.json',session,immutable=True)
    atomic_json(out/'ACCOUNTING_AT_COMPLETION.json',read_json(BASE/'runtime/ACCOUNTING.json'),immutable=True)
    result={'status':'COMPLETE','created_utc':stamp,'observed_timing':timing,
        'completed_jobs':completed,'preserved_resolved_jobs':resolved,'pending_required_jobs':0,
        'final_audit_sha256':digest(ROOT/'FINAL_AUDIT_NEXT4.json'),
        'delivery_receipt':pointer,'source_sha256':digest(Path(__file__)),
        'NEXT3_stopped_preserved':True,'all_seals_unopened':True,
        'full_project_archive':delivery['full_project_archive'],'twenty_file_folder':str(folder),
        'twenty_file_archive':delivery['twenty_file_archive']}
    atomic_json(out/'RESULTS.json',result,immutable=True)
    atomic_json(BASE/'SESSION.json',{**session,'status':'COMPLETE','completed_utc':stamp,
        'completion_receipt':'artifacts/next4/completion/RESULTS.json','completion_receipt_sha256':digest(out/'RESULTS.json'),
        'auto_resume_authorized':False,'final_session_outcome':audit['session_outcome']})
    print('NEXT4 SESSION COMPLETE: verified delivery and no required jobs remain',flush=True)


if __name__=='__main__':main()
