"""Pinned official point protocol with the consumer in a fresh Python process."""
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from src.validation.protocol import official_runner


def subprocess_protocol_predictions(series,model_directory):
    official=official_runner();command=official.RemoteCommand;predictions=[]
    root=Path(__file__).resolve().parents[2];server,endpoint=official.os_create_server_socket()
    env=os.environ.copy();env.pop('PYTHONPATH',None);env['PYTHONIOENCODING']='utf-8'
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
    started=time.perf_counter();child=None;timing={}
    try:
        with server:
            server.settimeout(60)
            child=subprocess.Popen([sys.executable,str(root/'scripts/official_socket_worker.py'),str(endpoint),str(Path(model_directory).resolve())],
                cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8',
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            client,_=official.os_accept(server)
            with client:
                client.settimeout(60);remote=official.Remote(client);remote.expect(command.READY)
                ready=time.perf_counter();timing['process_start_to_ready_seconds']=ready-started
                for historical,online in series:
                    h=np.asarray(historical,dtype=np.float32)
                    remote.send(command.NEW_TIMESERIES);remote.send(command.HISTORICAL_DATA,len(h));remote.send_raw(h.tobytes())
                    for point in online:
                        remote.send(command.ONLINE_POINT,float(point));predictions.append(float(remote.expect(command.NEW_PREDICTION)))
                    remote.send(command.END_TIMESERIES)
                remote.send(command.END);timing['ready_to_end_wire_seconds']=time.perf_counter()-ready
            stdout,stderr=child.communicate(timeout=60)
            if child.returncode!=0:raise RuntimeError('Official socket child failed; '+stderr[-2000:])
            if stdout.strip():raise RuntimeError('Official socket child emitted unexpected standard output')
        timing['process_total_seconds']=time.perf_counter()-started
        timing['consumer_exit_code']=child.returncode
        return np.asarray(predictions,dtype=np.float32),timing
    finally:
        if child is not None and child.poll() is None:
            child.kill();child.communicate(timeout=30)
