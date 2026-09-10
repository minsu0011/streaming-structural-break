"""Run unchanged official double-protection code over real loopback sockets."""
import importlib.util
import sys
import threading
from pathlib import Path
import numpy as np

def official_runner():
    path=Path(__file__).resolve().parents[2]/'vendor/official/runner.py'
    spec=importlib.util.spec_from_file_location('pinned_official_runner',path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module

def protocol_predictions(infer_function,series,model_directory):
    official=official_runner(); command=official.RemoteCommand
    errors=[]; predictions=[]
    server,endpoint=official.os_create_server_socket()
    with server:
        server.settimeout(30)
        def consumer():
            try:
                with official.os_create_client_socket() as sock:
                    sock.settimeout(30); official.os_connect(sock,endpoint)
                    official._run_with_double_protection(remote=official.Remote(sock),
                        consumer_factory=lambda datasets:infer_function(datasets,str(model_directory)),
                        post_processor=official._post_process_infer_yield_result)
            except BaseException as exc:
                errors.append(exc)
        thread=threading.Thread(target=consumer,daemon=True); thread.start()
        client,_=official.os_accept(server)
        with client:
            client.settimeout(30); remote=official.Remote(client)
            remote.expect(command.READY)
            for historical,online in series:
                h=np.asarray(historical,dtype=np.float32)
                remote.send(command.NEW_TIMESERIES)
                remote.send(command.HISTORICAL_DATA,len(h))
                remote.send_raw(h.tobytes())
                for point in online:
                    remote.send(command.ONLINE_POINT,float(point))
                    predictions.append(float(remote.expect(command.NEW_PREDICTION)))
                remote.send(command.END_TIMESERIES)
            remote.send(command.END)
        thread.join(timeout=30)
        if thread.is_alive(): raise TimeoutError('Official protocol consumer did not exit')
        if errors: raise errors[0]
    return np.asarray(predictions,dtype=np.float32)
