"""Bounded same-array native-crash investigation; never a model experiment."""
import argparse
import faulthandler
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def worker(mode,output):
    import numpy as np
    from src.scoring.ts_auc import ts_auc
    from src.next4.artifacts import BASE,read_json,digest,utc_now
    candidate=BASE/'synthetic/candidates/B2_LOG_ENERGY_SEMITAIL_all_dev/counterexample/CB_N3_070'
    reference=BASE/'synthetic/reference/all_dev/counterexample/CB_N3_070'
    evidence={}
    for directory in (candidate,reference):
        receipt=read_json(directory/'RESULTS.json');assert receipt['status']=='PASS'
        for name,sha in receipt['files_sha256'].items():
            assert digest(directory/name)==sha;evidence[(directory/name).relative_to(ROOT).as_posix()]=sha
    p=np.load(candidate/'predictions.npy',mmap_mode='r',allow_pickle=False)
    ref=np.load(reference/'BL80.npy',mmap_mode='r',allow_pickle=False)
    tau=np.load(reference/'tau.npy',allow_pickle=False);no=p.shape[1]
    times=np.broadcast_to(np.arange(no,dtype=np.int32),p.shape).ravel()
    target=((np.arange(no)[None,:]>=tau[:,None])&(tau[:,None]>=0)).astype(np.uint8).ravel()
    with Path(output).with_suffix('.stack.log').open('x',encoding='utf-8') as stream:
        faulthandler.enable(file=stream)
        if mode=='timed':faulthandler.dump_traceback_later(.03,repeat=False,file=stream)
        values=[]
        try:
            for _ in range(3):values.append([ts_auc(target,p.ravel(),times),ts_auc(target,ref.ravel(),times)])
        finally:faulthandler.cancel_dump_traceback_later()
    assert values[0]==values[1]==values[2]
    result={'status':'PASS','created_utc':utc_now(),'mode':mode,'same_array_score_pairs':values,
        'new_model_fits':0,'changed_model_or_predictions':False,'new_random_seeds':0,
        'python':sys.version,'GIL_enabled':sys._is_gil_enabled(),'evidence_sha256':evidence}
    with Path(output).open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',choices=['normal','timed']);parser.add_argument('--output');args=parser.parse_args()
    if args.worker:worker(args.worker,args.output);return
    from src.next4.artifacts import BASE,atomic_json,digest,utc_now
    out=BASE/'maintenance/SYN_NATIVE_FAILURE_001/probe';out.mkdir(parents=True,exist_ok=False);rows=[]
    for mode in ('normal','timed'):
        output=out/(mode+'.json')
        process=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--worker',mode,'--output',str(output)],cwd=ROOT,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),timeout=120)
        (out/(mode+'.log')).write_text(process.stdout,encoding='utf-8')
        rows.append({'mode':mode,'exit_code':process.returncode,'completed_receipt':output.exists()})
        print('NEXT4 NATIVE FAILURE PROBE',mode,process.returncode,flush=True)
    atomic_json(out/'RESULTS.json',{'status':'DIAGNOSTIC_COMPLETED','created_utc':utc_now(),'runs':rows,
        'source_sha256':digest(Path(__file__)),'new_fits':0,'new_bootstrap_draws':0,
        'inference_limit':'A successful short probe does not rule out an intermittent timed-stack crash. The original 1500s crash and its Windows event remain authoritative failure evidence.',
        'files_sha256':{p.name:digest(p) for p in out.iterdir()}},immutable=True)


if __name__=='__main__':main()
