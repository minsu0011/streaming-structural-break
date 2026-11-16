"""Expanded adversarial prefixes and all-DEV current-point API equivalence."""
from pathlib import Path
import sys
import hashlib
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
from src.next4.b2_inference import infer,load_prepared
from src.next4.cache import build_design
from src.next2.counterexamples import predict
from src.next2.blend import combine
from src.next.data import prepare_dev
from scripts.verify_next2_shared_public_api import Datasets


def main():
    policy_path=BASE/'diagnostic_policy/B2_RUNTIME_POLICY.json';policy=read_json(policy_path)
    for name,sha in policy['sources_sha256'].items():assert digest(ROOT/name)==sha
    directory=ROOT/policy['model_directory'];assert digest(directory/'model.joblib')==policy['model_sha256']
    out=BASE/'runtime_validation/B2_DIRECT';out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'START.json',{'created_utc':utc_now(),'policy_sha256':digest(policy_path),'new_fits':0},immutable=True)
    prepared=load_prepared(directory);rng=np.random.default_rng(policy['prefix_seed']);case_rows=[]
    for case in range(policy['prefix_cases']):
        nh=(192,256,512,1024)[case%4];n=64+case%97;prefix=1+case%(n-1)
        h=(rng.standard_t(5,size=nh)*(1+case%3)+case%7).astype(np.float32)
        a=rng.normal(size=n).astype(np.float32);b=a.copy();b[prefix:]=rng.normal(0,30,n-prefix)
        left=prepared.new_state(h);before=left.state_array_bytes
        p=np.asarray([left.predict_one(v) for v in a],dtype=np.float32)
        right=prepared.new_state(h);q=np.asarray([right.predict_one(v) for v in b],dtype=np.float32)
        np.testing.assert_array_equal(p[:prefix],q[:prefix])
        short=prepared.new_state(h);short_values=np.asarray([short.predict_one(v) for v in a[:prefix]],dtype=np.float32)
        np.testing.assert_array_equal(p[:prefix],short_values)
        x=prepared.model.make_state(h).replay(a)
        expected=combine(prepared.model.predict(x),predict(prepared.complement,h,a),.8)
        np.testing.assert_array_equal(p,expected);assert before==left.state_array_bytes
        case_rows.append({'case':case,'historical_length':nh,'online_length':n,'prefix':prefix,'state_array_bytes':before,'status':'PASS'})
        if (case+1)%25==0:
            progress('B2 expanded prefix invariance',str(case+1)+'/300 changed-suffix, shorter-horizon and batch/current-point cases')
            print('NEXT4 B2 PREFIX',case+1,flush=True)
    atomic_json(out/'PREFIX_CASES.json',{'status':'PASS','cases':case_rows,'new_fits':0},immutable=True)
    data=prepare_dev(ROOT);registration=read_json(BASE/'registrations/B2_LOG_ENERGY_SEMITAIL.json')
    x,metadata,names,rows,cache=build_design(data,registration,'full')
    datasets=Datasets(data,range(len(data.ids)));stream=infer(datasets,directory)
    assert datasets.accessed==0 and next(stream) is None and datasets.accessed==0
    expected_digest=hashlib.sha256();actual_digest=hashlib.sha256();primary_digest=hashlib.sha256()
    position=0;saved={};started=time.perf_counter();active=[False];writes=[]
    def audit(event,arguments):
        if not active[0]:return
        write=False
        if event=='open':
            import os
            path,mode,flags=arguments
            write=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if write and isinstance(path,(str,bytes,os.PathLike)) and os.path.normcase(os.path.normpath(os.fsdecode(path)))==os.path.normcase(os.path.normpath(os.devnull)):return
        if event in ('os.mkdir','os.remove','os.rmdir','os.rename','os.chmod','os.truncate','os.link','os.symlink'):write=True
        if write:writes.append(event);raise RuntimeError('B2 direct inference attempted persistent filesystem mutation')
    sys.addaudithook(audit)
    for j,sid in enumerate(data.ids):
        h,o,_=data.get(int(sid));primary=np.asarray(prepared.model.predict(x[position:position+len(o)]),dtype=np.float32)
        expected=combine(primary,predict(prepared.complement,h,o),.8);actual=np.empty(len(o),dtype=np.float32)
        active[0]=True
        try:
            for k in range(len(o)):
                value=next(stream)
                assert type(value) is float and np.isfinite(value) and 0<=value<=1
                assert datasets.index==j and datasets.current.consumed==k+1
                actual[k]=value
        finally:active[0]=False
        np.testing.assert_array_equal(actual,expected)
        primary_digest.update(primary.tobytes());expected_digest.update(expected.tobytes());actual_digest.update(actual.tobytes())
        if j<20:saved[j]=actual.copy()
        position+=len(o)
        if (j+1)%250==0:
            progress('B2 direct API all-DEV parity',str(j+1)+'/8000; one current scalar per output, cached feature/causal C205 oracle exact')
            print('NEXT4 B2 DIRECT ALL DEV',j+1,flush=True)
    try:next(stream)
    except StopIteration:pass
    else:raise AssertionError('Extra B2 API prediction')
    assert expected_digest.hexdigest()==actual_digest.hexdigest()
    assert primary_digest.hexdigest()==read_json(directory/'RESULTS.json')['all_DEV_prediction_sha256']
    reordered=Datasets(data,list(reversed(saved)));repeat=infer(reordered,directory)
    assert next(repeat) is None and reordered.accessed==0
    for j in reversed(saved):
        values=np.asarray([next(repeat) for _ in range(len(saved[j]))],dtype=np.float32)
        np.testing.assert_array_equal(values,saved[j])
    try:next(repeat)
    except StopIteration:pass
    else:raise AssertionError('Extra reordered B2 prediction')
    assert not writes;data.guard.check_state()
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'candidate':'B2_LOG_ENERGY_SEMITAIL','promotion':False,
        'prefix_cases':len(case_rows),'all_DEV_series':len(data.ids),'all_DEV_points':position,
        'output_fingerprint_sha256':actual_digest.hexdigest(),'expected_fingerprint_sha256':expected_digest.hexdigest(),
        'primary_refit_fingerprint_exact':True,'reordered_reset_series':len(saved),
        'first_none_before_any_dataset_access':True,'current_point_consumption_exact':True,
        'future_length_and_suffix_inaccessible':True,'persistent_inference_write_attempts':len(writes),
        'state_array_bytes_min':min(r['state_array_bytes'] for r in case_rows),'state_array_bytes_max':max(r['state_array_bytes'] for r in case_rows),
        'new_fits':0,'seal_rows':0,'real_in_sample_quality_score_computed':False,'fresh_raw_reproduction':False,
        'elapsed_all_DEV_seconds':time.perf_counter()-started,'policy_sha256':digest(policy_path),
        'files_sha256':{'PREFIX_CASES.json':digest(out/'PREFIX_CASES.json')}},immutable=True)
    print('NEXT4 B2 DIRECT VERIFICATION COMPLETE',flush=True)


if __name__=='__main__':main()
