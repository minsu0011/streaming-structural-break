"""Verify an additive import-only repair with cold writes blocked from import."""
from pathlib import Path
import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(output):
    import numpy as np
    from scripts.verify_next4_cold_api import Online,Datasets
    assert sys.dont_write_bytecode
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
        if write:writes.append(event);raise RuntimeError('Read-only B2 entry attempted persistent mutation: '+event)
    sys.addaudithook(audit);active[0]=True;started=time.perf_counter();fingerprint=hashlib.sha256();points=0
    try:
        entry=importlib.import_module('next4_b2_readonly_diagnostic_main')
        core=importlib.import_module('src.next4.b2_inference')
        assert entry.infer is core.infer
        directory=ROOT/'artifacts/next4/refits/B2_LOG_ENERGY_SEMITAIL/diagnostic_all_dev_v1'
        data=Datasets(records);iterator=entry.infer(data,directory)
        assert data.index==-1 and next(iterator) is None and data.index==-1
        saved=[]
        for j,(h,o) in enumerate(records):
            values=np.empty(len(o),dtype=np.float32)
            for k in range(len(o)):
                value=next(iterator);assert type(value) is float and np.isfinite(value) and 0<=value<=1
                assert data.index==j and data.current.consumed==k+1;values[k]=value
            saved.append(values);fingerprint.update(values.tobytes());points+=len(o)
        try:next(iterator)
        except StopIteration:pass
        else:raise AssertionError('Unexpected extra output')
        # Same exact core function with a separately loaded prepared object.
        prepared=core.load_prepared(directory)
        for (h,o),expected in zip(records,saved):
            state=prepared.new_state(h)
            actual=np.array([state.predict_one(v) for v in o],dtype=np.float32)
            np.testing.assert_array_equal(actual,expected)
        from src.next.readonly_bootstrap import cache_requests_suppressed
        suppressed=cache_requests_suppressed();assert suppressed
    except BaseException as exc:
        active[0]=False
        Path(output).with_suffix('.failure.json').write_text(json.dumps({'status':'FAIL','exception_type':type(exc).__name__,
            'message':str(exc),'traceback':traceback.format_exc(),'write_attempts':writes},indent=2)+'\n',encoding='utf-8')
        raise
    finally:active[0]=False
    result={'status':'PASS','created_utc':datetime.now(timezone.utc).isoformat(),'entry':'next4_b2_readonly_diagnostic_main.py',
        'entry_sha256':sha(ROOT/'next4_b2_readonly_diagnostic_main.py'),
        'core_infer_is_same_function_object':True,'independent_prepared_core_predictions_exact':True,
        'cold_import_write_attempts':len(writes),'audit_enabled_before_model_entry_import':True,
        'interpreter_bytecode_writes_disabled':True,'fresh_process':True,'datasets':len(records),'points':points,
        'prediction_sha256':fingerprint.hexdigest(),'suppressed_project_cache_requests':len(suppressed),
        'first_none_before_data':True,'current_point_consumption_exact':True,'new_fits':0,'real_or_seal_rows':0,
        'cold_import_compile_and_infer_seconds':time.perf_counter()-started}
    with Path(output).open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker-output');args=parser.parse_args()
    if args.worker_output:worker(args.worker_output);return
    base=ROOT/'artifacts/next4';policy=json.loads((base/'diagnostic_policy/B2_READONLY_ENTRY_V2.json').read_text())
    for name,value in policy['source_sha256'].items():assert sha(ROOT/name)==value
    for name,value in policy['unchanged_scientific_source_sha256'].items():assert sha(ROOT/name)==value
    out=base/'cold_api_readonly_v2';out.mkdir(parents=True,exist_ok=False);results=[]
    for repeat in (1,2):
        path=out/f'B2_{repeat}.json'
        process=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--worker-output',str(path)],cwd=ROOT,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        (out/f'B2_{repeat}.log').write_text(process.stdout,encoding='utf-8');process.check_returncode()
        results.append(json.loads(path.read_text()));print('NEXT4 READONLY B2 COLD API',repeat,'PASS',flush=True)
    assert len({r['prediction_sha256'] for r in results})==1
    result={'status':'PASS','created_utc':datetime.now(timezone.utc).isoformat(),'fresh_B2_processes':2,
        'all_repeated_predictions_exact':True,'new_fits':0,'real_or_seal_rows':0,'new_candidate_identity':False,
        'policy_sha256':sha(base/'diagnostic_policy/B2_READONLY_ENTRY_V2.json'),'results':results,
        'original_failure_retained':'artifacts/next4/jobs/CI001_COLD_API_DETERMINISM/attempt_1/RUN.log',
        'failure_trace':'artifacts/next4/maintenance/COLD_API_FAILURE_001/EXCEPTION.json',
        'original_direct_timing_scope':'Earlier five paired timings measured the unchanged prepared scalar core. This adapter alters import-time cache IO only and has no per-point wrapper or model changes.',
        'files_sha256':{p.name:sha(p) for p in out.iterdir()}}
    with (out/'RESULTS.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print('NEXT4 READONLY B2 COLD API COMPLETE',flush=True)


if __name__=='__main__':main()
