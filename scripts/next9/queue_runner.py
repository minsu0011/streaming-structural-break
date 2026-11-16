"""Owned subprocess wrapper; completion survives scheduler crashes."""
from pathlib import Path
import argparse
import os
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.next9.common import BASE,read_json,save,digest,utc_now
from src.next9.worker_environment import environment,THREAD_KEYS


def main(taskid,attempt):
    directory=BASE/f'runtime/queue/{taskid}/attempt_{attempt}'
    item=read_json(directory/'TASK.json')
    script=(ROOT/item['script']).resolve()
    assert script.is_relative_to(ROOT/'scripts/next9') and digest(script)==item['source_sha256']
    with (directory/'child.log').open('a',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,'-X','utf8','-B',str(script),*item.get('arguments',[])],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            env=environment(),
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    save(str((directory/'DONE.json').relative_to(BASE)),{'status':'COMPLETE' if result.returncode==0 else 'FAILED',
        'timestamp':utc_now(),'returncode':result.returncode,'task_id':taskid,'attempt':attempt,
        'source_sha256':item['source_sha256'],'partial_artifacts_never_deleted':True})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--task-id',required=True)
    parser.add_argument('--attempt',type=int,required=True)
    args=parser.parse_args()
    main(args.task_id,args.attempt)
