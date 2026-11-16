"""Two independent cold, read-only processes for an actually nominated model."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',required=True);args=parser.parse_args();cid=args.candidate
    nomination=read_json(BASE/'finalist_nominations'/(cid+'.json'))
    for name,sha in nomination['execution_sources_sha256'].items():assert digest(ROOT/name)==sha
    out=BASE/'finalist_validation'/cid/'COLD';out.mkdir(parents=True,exist_ok=False);records=[]
    for repeat in (1,2):
        target=out/f'cold_{repeat}.json'
        environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        p=subprocess.run([sys.executable,'-B','scripts/verify_next4_candidate_api.py','--model',cid,'--cold-worker-output',str(target)],
            cwd=ROOT,env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),timeout=240)
        (out/f'cold_{repeat}.log').write_text(p.stdout,encoding='utf-8');p.check_returncode()
        r=read_json(target);assert r['status']=='PASS' and r['cold_import_write_attempts']==0;records.append(r)
        progress('Nominated robustness cold API',cid+' repeat '+str(repeat)+' cold entry import and current-only scalar inference passed')
        print('NEXT4 ROBUST COLD',cid,repeat,'PASS',flush=True)
    assert records[0]['prediction_sha256']==records[1]['prediction_sha256']
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'candidate':cid,'records':records,
        'fresh_processes':2,'all_repeated_predictions_exact':True,'new_fits':0,'seal_rows':0,
        'files_sha256':{p.name:digest(p) for p in out.iterdir() if p.is_file()}},immutable=True)


if __name__=='__main__':main()
