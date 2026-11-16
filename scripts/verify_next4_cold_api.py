"""Two fresh Python processes test cold B2/reference inference without writes."""
from pathlib import Path
import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Online:
    def __init__(self,values):self.values=values;self.consumed=0
    def __len__(self):raise AssertionError('Future length access')
    def __array__(self,*args,**kwargs):raise AssertionError('Future suffix materialization')
    def __iter__(self):
        for value in self.values:
            self.consumed+=1;yield float(value)


class Datasets:
    def __init__(self,records):self.records=records;self.index=-1;self.current=None
    def __iter__(self):
        for index,(h,o) in enumerate(self.records):
            self.index=index;self.current=Online(o);yield h,self.current


def worker(label,target):
    import numpy as np
    assert sys.dont_write_bytecode,'Fresh process must disable interpreter bytecode writes'
    if label=='B2':entry_name='next4_b2_diagnostic_main';directory=ROOT/'artifacts/next4/refits/B2_LOG_ENERGY_SEMITAIL/diagnostic_all_dev_v1';model_name='model.joblib'
    else:entry_name='next2_shared_performance_main';directory=ROOT/'artifacts/next2/canonical_deployment/PERFORMANCE_shared_v1';model_name='model.sbnext2'
    records=[]
    for index in range(20):
        htime=np.arange(192+index%4*64,dtype=np.float64);otime=np.arange(80+index*3,dtype=np.float64)
        h=(np.sin(.37*htime)+.2*np.cos(.11*htime)+.03*index).astype(np.float32)
        o=(np.sin(.37*(otime+len(h)))+.2*np.cos(.11*(otime+len(h)))+.03*index).astype(np.float32)
        records.append((h,o))
    active=[False];writes=[]
    def audit(event,arguments):
        if not active[0]:return
        write=False
        if event=='open':
            path,mode,flags=arguments
            write=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if write and isinstance(path,(str,bytes,os.PathLike)) and os.path.normcase(os.path.normpath(os.fsdecode(path)))==os.path.normcase(os.path.normpath(os.devnull)):return
        if event in ('os.mkdir','os.remove','os.rmdir','os.rename','os.chmod','os.truncate','os.link','os.symlink'):write=True
        if write:writes.append(event);raise RuntimeError('Cold model import/inference attempted persistent mutation: '+event)
    sys.addaudithook(audit);active[0]=True;started=time.perf_counter();fingerprint=hashlib.sha256();points=0
    try:
        entry=importlib.import_module(entry_name);data=Datasets(records);iterator=entry.infer(data,directory)
        assert data.index==-1 and next(iterator) is None and data.index==-1
        for j,(h,o) in enumerate(records):
            output=np.empty(len(o),dtype=np.float32)
            for k in range(len(o)):
                value=next(iterator);assert np.isscalar(value) and np.isfinite(value) and 0<=value<=1
                assert data.index==j and data.current.consumed==k+1;output[k]=value
            fingerprint.update(output.tobytes());points+=len(o)
        try:next(iterator)
        except StopIteration:pass
        else:raise AssertionError('Unexpected extra output')
    finally:active[0]=False
    result={'status':'PASS','created_utc':datetime.now(timezone.utc).isoformat(),'model':label,
        'entry_sha256':sha(ROOT/(entry_name+'.py')),'model_sha256':sha(directory/model_name),
        'fresh_process':True,'interpreter_bytecode_writes_disabled':True,'audit_active_before_entry_import':True,
        'prepared_model_or_kernel_warmed_before_write_audit':False,'persistent_write_attempts':len(writes),
        'datasets':len(records),'points':points,'first_none_before_data':True,'one_current_point_per_output':True,
        'future_length_and_materialization_traps':True,'prediction_sha256':fingerprint.hexdigest(),
        'cold_import_compile_and_infer_seconds':time.perf_counter()-started,'new_fits':0,'real_or_seal_rows':0}
    with Path(target).open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',choices=['B2','BL80']);parser.add_argument('--output');args=parser.parse_args()
    if args.worker:worker(args.worker,args.output);return
    out=ROOT/'artifacts/next4/cold_api';out.mkdir(parents=True,exist_ok=False)
    results=[]
    for label in ('BL80','B2'):
        for repeat in (1,2):
            path=out/(label+'_'+str(repeat)+'.json')
            subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--worker',label,'--output',str(path)],cwd=ROOT,
                check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            results.append(json.loads(path.read_text(encoding='utf-8')))
            print('NEXT4 COLD API',label,repeat,'PASS',flush=True)
    for label in ('BL80','B2'):
        selected=[r for r in results if r['model']==label];assert len({r['prediction_sha256'] for r in selected})==1
    result={'status':'PASS','created_utc':datetime.now(timezone.utc).isoformat(),'fresh_processes':4,
        'per_model_repeats':2,'all_repeated_predictions_exact':True,'write_audit_covers_cold_entry_import':True,
        'interpreter_bytecode_writes_explicitly_disabled':True,'new_fits':0,'real_or_seal_rows':0,
        'source_sha256':sha(Path(__file__)),'results':results,
        'scope':'Deterministic feature-input smoke sequences in fresh Python processes. This supplements the warmed all-DEV exact replay; it is not new quality evidence or raw reproduction.',
        'files_sha256':{p.name:sha(p) for p in out.glob('*.json')}}
    with (out/'RESULTS.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print('NEXT4 COLD API PROCESS DETERMINISM COMPLETE',flush=True)


if __name__=='__main__':main()
