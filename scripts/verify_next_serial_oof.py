"""All-five-fold guarded API replay with read-only inference and exact OOFs."""
import os
import sys
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import argparse
import json
import shutil
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.data import prepare_dev
from src.next.candidate_io import load_full_oof
from src.next.serial_inference import implementation_hash
from src.next.serial_variance_runtime import source_hash
from scripts.verify_next_serial_contract import run_guarded
from src.utils.artifacts import write_json,sha256,utc_now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate',default='J102_SERIAL_VARIANCE_512')
    args = parser.parse_args()
    data = prepare_dev(ROOT)
    candidate,prior,metadata,expected,sources = load_full_oof(data,args.candidate)
    out = ROOT/'artifacts/next/inference_oof'/args.candidate
    out.mkdir(parents=True,exist_ok=True)
    identity = {'candidate':candidate,'oof_sources':sources,'inference_sha256':implementation_hash(),
        'runtime_sha256':source_hash(),'script_sha256':sha256(Path(__file__)),
        'guard_helper_sha256':sha256(ROOT/'scripts/verify_next_serial_contract.py')}
    if (out/'POLICY.json').exists():
        if json.loads((out/'POLICY.json').read_text(encoding='utf-8'))!=identity:
            raise RuntimeError('Full API replay identity changed')
    else:
        write_json(out/'POLICY.json',identity)
    active = [False]
    def audit(event,args):
        if not active[0]:
            return
        if event=='open':
            path,mode,flags = args
            if (isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)):
                raise AssertionError('Full API replay attempted a file write')
        if event in ('os.mkdir','os.remove','os.rename','os.rmdir','os.chmod','os.truncate','os.link','os.symlink'):
            raise AssertionError('Full API replay attempted a filesystem mutation')
    sys.addaudithook(audit)
    records = []
    for fold in range(5):
        path = out/f'fold_{fold}.json'
        if path.exists():
            record = json.loads(path.read_text(encoding='utf-8'))
            if record['status']!='PASS':
                raise RuntimeError('A previous full API fold failed')
            records.append(record)
            continue
        folder = out/f'fold_{fold}_artifact'
        folder.mkdir(exist_ok=True)
        source = ROOT/'artifacts/next/experiments'/args.candidate/'full'/f'fold_{fold}.joblib'
        descriptor = {'kind':'NEXT_REGISTERED_BINARY','runtime':'serial_variance_fast_h','candidate':candidate,
            'model_sha256':sha256(source),'inference_sha256':implementation_hash(),'runtime_sha256':source_hash(),
            'status':'FULL_PRIMARY_OOF_API_VERIFICATION_ONLY','fold':fold}
        if (folder/'model.json').exists():
            if json.loads((folder/'model.json').read_text(encoding='utf-8'))!=descriptor or sha256(folder/'model.joblib')!=descriptor['model_sha256']:
                raise RuntimeError('Full API replay fixture changed')
        else:
            shutil.copy2(source,folder/'model.joblib')
            write_json(folder/'model.json',descriptor)
        examples = []
        for sid in data.ids[data.series['fold']==fold]:
            h,o,_ = data.get(sid)
            examples.append((h,o,len(o)))
        tick = time.perf_counter()
        active[0] = True
        try:
            predictions = run_guarded(examples,folder)
        finally:
            active[0] = False
        actual = np.concatenate(predictions)
        rows = np.flatnonzero(metadata['fold']==fold)
        np.testing.assert_array_equal(actual,expected[rows])
        record = {'status':'PASS','fold':fold,'series':len(examples),'points':len(actual),
            'all_float32_predictions_bitwise_equal':True,'single_pass_and_no_read_ahead':True,
            'readiness_consumes_zero_inputs':True,'inference_file_writes':0,'maximum_absolute_difference':0.,
            'elapsed_seconds':time.perf_counter()-tick,'created_utc':utc_now(),'seal_rows':0}
        write_json(path,record)
        records.append(record)
        print('NEXT FULL API PARITY',args.candidate,fold,len(actual),round(record['elapsed_seconds'],1),flush=True)
    if sum(record['points'] for record in records)!=4014405 or sum(record['series'] for record in records)!=8000:
        raise RuntimeError('Full API replay coverage incomplete')
    write_json(out/'RESULTS.json',{'status':'PASS','candidate':args.candidate,'identity':identity,'folds':records,
        'series':8000,'points':4014405,'all_float32_predictions_bitwise_equal':True,
        'inference_file_writes':0,'seal_rows':0,'reduced_new_usage':0,'created_utc':utc_now(),
        'scope':'Every original DEV OOF point through the separate NEXT infer generator, with guarded outer/inner iterators and Python audit-hook write denial. Module import precedes the inference guard; no cloud claim.'})
    print('NEXT FULL API PARITY COMPLETE',args.candidate,flush=True)


if __name__=='__main__':
    main()
