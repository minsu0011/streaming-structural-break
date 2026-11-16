"""Run full pinned official run()/execute() with fresh subprocesses and synthetic parquet.

Authentication and cloud sandbox are not emulated. This is explicitly not
crunch_tools.test() or an official dataset score.
"""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.validation.protocol import official_runner
from src.validation.synthetic import fixture,KINDS
from src.data.labels import labels_from_tau
from src.streaming.inference import infer
from src.utils.artifacts import write_json,utc_now,ledger
from scripts.verify_scorer import load_official

def main():
    started=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='structural_break_official_smoke_') as temp:
        base=Path(temp);data=base/'data';model=base/'model';prediction=base/'prediction'
        for directory in (data,model,prediction):directory.mkdir()
        frames=[];targets=[];indices=[];series=[]
        for sid in range(20):
            h,o,tau=fixture(KINDS[sid%len(KINDS)],seed=20260908+sid,historical_length=1000,online_length=120,break_at=40)
            keys=pd.MultiIndex.from_product([[sid],range(100,100+len(h)+len(o))],names=['id','time'])
            frames.append(pd.DataFrame({'value':np.r_[h,o],'period':np.r_[np.ones(len(h),dtype=np.int8),np.full(len(o),2,dtype=np.int8)]},index=keys))
            targets.append(pd.DataFrame({'target':labels_from_tau(len(o),tau)},index=keys[len(h):]))
            indices.append({'id':sid,'tau_index':-1 if tau is None else tau})
            series.append((h,o))
        x=pd.concat(frames);y=pd.concat(targets)
        x.to_parquet(data/'X_train.parquet');x.to_parquet(data/'X_test.reduced.parquet')
        y.to_parquet(data/'y_test.parquet');pd.DataFrame(indices).set_index('id').to_parquet(data/'y_train_index.parquet')
        logs=[];calls=[];determinism=[]
        class Context:
            is_local=True
            force_first_train=True
            is_determinism_check_enabled=True
            def execute(self,command,parameters=None,**kwargs):
                env=os.environ.copy()
                for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):env[name]='1'
                result=subprocess.run([sys.executable,str(ROOT/'scripts/offline_runner_worker.py'),'--command',command,'--data',str(data),'--model',str(model),'--parameters',json.dumps(parameters or {})],
                    cwd=ROOT,env=env,text=True,capture_output=True,timeout=60)
                calls.append({'command':command,'exit_code':result.returncode})
                logs.append(result.stdout+result.stderr)
                if result.returncode:raise RuntimeError(result.stderr)
            def log(self,message,**kwargs):logs.append(message)
            def report_determinism(self,value):determinism.append(bool(value))
        class Tracer:
            def span(self,*args,**kwargs):return contextlib.nullcontext()
        official_runner().run(Context(),Tracer(),str(data),str(prediction))
        official_scoring=load_official()
        with contextlib.redirect_stdout(io.StringIO()):
            official_scoring.check(str(prediction),str(data))
        output=pd.read_parquet(prediction/'prediction.parquet')
        direct=np.asarray(list(infer(iter(series),model))[1:],dtype=np.float32)
        difference=float(np.max(abs(output.prediction.to_numpy()-direct)))
        report={'status':'PASS','timestamp':utc_now(),'scope':'SYNTHETIC_FULL_OFFICIAL_RUNNER_SUBPROCESS',
            'training_function':'Explicit synthetic engineering baseline writer; production refit has separate DEV-only contract verification',
            'official_run_function':True,'official_execute_function':True,'official_prediction_check':True,
            'subprocess_calls':calls,'rows':len(output),'expected_rows':len(y),'index_names':output.index.names,
            'indices_match':output.index.equals(y.index),'prediction_dtype':str(output.prediction.dtype),'finite':bool(np.isfinite(output.prediction).all()),
            'range':[float(output.prediction.min()),float(output.prediction.max())],'official_10_percent_determinism':determinism,
            'direct_vs_wire_max_abs_diff':difference,'elapsed_seconds':time.perf_counter()-started,
            'authenticated_crunch_tools_test':False,'cloud_sandbox_validated':False,'competition_data_used':False,'reduced_scored_uses':0}
        assert len(output)==len(y) and output.index.equals(y.index) and determinism==[True] and difference==0
        write_json(ROOT/'artifacts/validation/FULL_OFFICIAL_RUNNER_SMOKE.json',report)
        (ROOT/'experiments/logs/full_official_runner.log').write_text('\n'.join(logs),encoding='utf-8')
        ledger(experiment_id='full_official_runner_subprocess',model='B7_manual',status='ENGINEERING_PASS',notes='Synthetic parquet; full official run/execute/check with new processes; no authentication or competition score')
        print(report)

if __name__=='__main__':main()
