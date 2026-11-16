"""Raw-free public-entry replay of the three historical frozen reference models."""
from pathlib import Path
import argparse
import hashlib
import importlib
import os
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode=True
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next7/numba_cache')
from src.next7.worker_environment import environment
os.environ.update(environment())
import numpy as np
from src.next7.common import BASE,read_json,save,digest,utc_now

CASES=(('N0',11),('V4',1),('RETURN_NORMAL',0))


def calculate():
    lock=read_json(BASE/'FROZEN_REFERENCE_DELIVERY_LOCK_NEXT7.json')
    for name,sha in lock['files_sha256'].items():assert digest(ROOT/name)==sha,name
    from src.next6_extension.synthetic import generated
    suite=read_json(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json')
    predictions,inputs={},{}
    for role in lock['references']:
        entry=importlib.import_module(Path(role['entry']).stem)
        for case,seed in CASES:
            h,online,_,_=generated(case,seed,partition=31,horizon=1024,control=False,lock=suite)
            key=role['id']+'_'+case+'_seed'+str(seed)
            output=entry.infer(iter([(h,iter(online))]),str(ROOT/role['model_directory']))
            assert next(output) is None
            values=np.fromiter(output,dtype=np.float64)
            assert values.shape==(1024,) and np.isfinite(values).all() and np.all((values>=0)&(values<=1))
            predictions[key]=values
            inputs[case+'_seed'+str(seed)]={'H':hashlib.sha256(h.tobytes()).hexdigest(),
                                          'O':hashlib.sha256(online.tobytes()).hexdigest()}
    return predictions,inputs


def main(output_directory,deny_original_root):
    target=Path(output_directory).resolve()
    assert not target.exists(),'Fresh reproduction output required'
    denied=[]
    original=Path(deny_original_root).resolve() if deny_original_root else None
    if original is not None:assert original!=ROOT.resolve()
    allowed_environment=(original/'.venv').resolve() if original else None
    def guard(event,arguments):
        if event!='open' or not arguments or not isinstance(arguments[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(arguments[0])).resolve()
        if path.is_relative_to((ROOT/'data').resolve()):
            denied.append(str(path));raise PermissionError('Frozen reference replay cannot read competition data')
        if original is not None and path.is_relative_to(original):
            if path.is_relative_to(ROOT.resolve()) or path.is_relative_to(target) or path.is_relative_to(allowed_environment):return
            denied.append(str(path));raise PermissionError('Original project read blocked for frozen reference replay')
    sys.addaudithook(guard)
    fixture_path=BASE/'frozen_reference_fixture/FIXTURE.json'
    fixture=read_json(fixture_path)
    assert fixture['reproducer_sha256']==digest(__file__)
    assert digest(BASE/'FROZEN_REFERENCE_DELIVERY_LOCK_NEXT7.json')==fixture['reference_lock_sha256']
    source=BASE/'frozen_reference_fixture/PREDICTIONS.npz'
    assert digest(source)==fixture['predictions_sha256']
    predictions,inputs=calculate()
    assert inputs==fixture['synthetic_input_sha256']
    checks=[]
    with np.load(source,allow_pickle=False) as expected:
        assert set(expected.files)==set(predictions)
        for name,values in predictions.items():
            np.testing.assert_array_equal(values,expected[name])
            checks.append({'path':name,'points':len(values),'bitwise_equal':True,'max_absolute_error':0.})
    assert len(checks)==9 and not denied
    target.mkdir(parents=True,exist_ok=False)
    from src.next7.common import atomic
    atomic(target/'FROZEN_REFERENCE_REPRODUCTION_DONE.json',{'status':'PASS','created_utc':utc_now(),
        'source_sha256':digest(__file__),'fixture_sha256':digest(fixture_path),
        'reference_lock_sha256':digest(BASE/'FROZEN_REFERENCE_DELIVERY_LOCK_NEXT7.json'),
        'public_entry_paths':9,'scalar_predictions':9216,'checks':checks,
        'original_workspace_access_blocked':original is not None,'denied_access_attempts':len(denied),
        'competition_observation_reads':0,'new_fits':0,'new_scores':0,
        'scope':'Three historical frozen reference models via their public infer generators. This checks packaging/inference identity, not RP OOF performance or a fresh full-DEV fit.'})
    print('FROZEN_REFERENCE_REPRODUCTION PASS 9 paths 9216 exact scalar outputs',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output-directory',required=True)
    parser.add_argument('--deny-original-root')
    args=parser.parse_args();main(args.output_directory,args.deny_original_root)
