"""Fresh complete scalar/API, prefix and serialization checks with bounded models."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def specification(label):
    if label=='BL80':return 'next2_shared_performance_main',ROOT/'artifacts/next2/canonical_deployment/PERFORMANCE_shared_v1'
    if label=='B2':return 'next4_b2_readonly_diagnostic_main',ROOT/'artifacts/next4/refits/B2_LOG_ENERGY_SEMITAIL/diagnostic_all_dev_v1'
    from src.next4.robust_finalist import ALLOWED
    if label not in ALLOWED:raise ValueError('Existing bounded model required')
    return 'next4_robustness_main',ROOT/'artifacts/next4/refits'/label/'nominated_all_dev_v1'


def write_audit(active,writes):
    def audit(event,arguments):
        if not active[0]:return
        write=False
        if event=='open':
            path,mode,flags=arguments
            write=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if write and isinstance(path,(str,bytes,os.PathLike)) and os.path.normcase(os.path.normpath(os.fsdecode(path)))==os.path.normcase(os.path.normpath(os.devnull)):return
        if event in ('os.mkdir','os.remove','os.rmdir','os.rename','os.chmod','os.truncate','os.link','os.symlink'):write=True
        if write:writes.append(event);raise RuntimeError('Candidate inference attempted persistent mutation: '+event)
    return audit


def cold_worker(label,output):
    import numpy as np
    from scripts.verify_next4_cold_api import Datasets
    assert sys.dont_write_bytecode
    # Do not import the model implementation before enabling the cold write audit.
    if label in ('BL80','B2'):entry_name,directory=specification(label)
    else:
        assert re.fullmatch('[A-Za-z0-9_]+',label)
        entry_name='next4_robustness_main';directory=ROOT/'artifacts/next4/refits'/label/'nominated_all_dev_v1'
    records=[]
    for index in range(20):
        ht=np.arange(192+index%4*64,dtype=np.float64);ot=np.arange(80+index*3,dtype=np.float64)
        h=(np.sin(.37*ht)+.2*np.cos(.11*ht)+.03*index).astype(np.float32)
        o=(np.sin(.37*(ot+len(h)))+.2*np.cos(.11*(ot+len(h)))+.03*index).astype(np.float32);records.append((h,o))
    active=[True];writes=[];sys.addaudithook(write_audit(active,writes));started=time.perf_counter();fingerprint=hashlib.sha256();points=0
    try:
        entry=importlib.import_module(entry_name);datasets=Datasets(records);stream=entry.infer(datasets,directory)
        assert datasets.index==-1 and next(stream) is None and datasets.index==-1
        for j,(h,o) in enumerate(records):
            values=np.empty(len(o),dtype=np.float32)
            for k in range(len(o)):
                value=next(stream);assert np.isscalar(value) and np.isfinite(value) and 0<=value<=1
                assert datasets.index==j and datasets.current.consumed==k+1;values[k]=value
            fingerprint.update(values.tobytes());points+=len(o)
        try:next(stream)
        except StopIteration:pass
        else:raise AssertionError('Extra cold API output')
    finally:active[0]=False
    result={'status':'PASS','created_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        'model':label,'entry':entry_name+'.py','prediction_sha256':fingerprint.hexdigest(),'points':points,'series':20,
        'cold_import_write_attempts':len(writes),'audit_enabled_before_entry_import':True,'first_none_before_data':True,
        'current_point_consumption_exact':True,'future_length_and_suffix_traps':True,'fresh_process':True,
        'elapsed_seconds':time.perf_counter()-started,'new_fits':0,'real_or_seal_rows':0}
    with Path(output).open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True);parser.add_argument('--tag');parser.add_argument('--audit',action='store_true');parser.add_argument('--cold-worker-output');args=parser.parse_args()
    if args.cold_worker_output:return cold_worker(args.model,args.cold_worker_output)
    import numpy as np
    from src.next4.artifacts import BASE,read_json,atomic_json,digest,utc_now,progress
    from src.next.data import prepare_dev
    from src.next2.counterexamples import predict
    from src.next2.blend import combine
    from scripts.verify_next2_shared_public_api import Datasets
    assert args.tag and re.fullmatch('[A-Za-z0-9_]+',args.tag)
    if args.audit:assert read_json(BASE/'SESSION.json').get('audit_started_utc')
    phase='FINAL_AUDIT' if args.audit else 'RESEARCH'
    entry_name,directory=specification(args.model);entry=importlib.import_module(entry_name)
    if args.model=='BL80':
        from src.next4.synthetic import load_reference
        prepared,_=load_reference('all_dev')
    elif args.model=='B2':
        from src.next4.b2_inference import load_prepared
        prepared=load_prepared(directory)
    else:
        from src.next4.robust_finalist import load_prepared
        prepared=load_prepared(directory)
    out=BASE/('final_audit' if args.audit else 'finalist_validation')/args.tag/('API_'+args.model);out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'START.json',{'created_utc':utc_now(),'model':args.model,'source_sha256':digest(Path(__file__)),
        'entry_sha256':digest(ROOT/(entry_name+'.py')),'audit_phase':args.audit},immutable=True)
    def expected(h,o,features=None):
        if args.model=='BL80':return combine(predict(prepared.primary,h,o),predict(prepared.complement,h,o),.8)
        x=prepared.model.make_state(h).replay(o) if features is None else features
        expert=np.asarray(prepared.model.predict(x),dtype=np.float32)
        if args.model=='B2':return combine(expert,predict(prepared.complement,h,o),.8)
        ref=predict(prepared.reference,h,o)
        return combine(expert,ref,.8) if prepared.augmented else combine(ref,expert,.9)
    rng=np.random.default_rng(20260908);cases=[]
    for case in range(300):
        nh=(192,256,512,1024)[case%4];n=64+case%97;prefix=1+case%(n-1)
        h=(rng.standard_t(5,size=nh)*(1+case%3)+case%7).astype(np.float32)
        a=rng.normal(size=n).astype(np.float32);b=a.copy();b[prefix:]=rng.normal(0,30,n-prefix)
        state=prepared.new_state(h);size=state.state_array_bytes;p=np.asarray([state.predict_one(v) for v in a],dtype=np.float32)
        other=prepared.new_state(h);q=np.asarray([other.predict_one(v) for v in b],dtype=np.float32)
        short=prepared.new_state(h);r=np.asarray([short.predict_one(v) for v in a[:prefix]],dtype=np.float32)
        np.testing.assert_array_equal(p[:prefix],q[:prefix]);np.testing.assert_array_equal(p[:prefix],r)
        np.testing.assert_array_equal(p,expected(h,a));assert size==state.state_array_bytes
        cases.append({'case':case,'status':'PASS','prefix':prefix,'state_array_bytes':size})
        if (case+1)%25==0:
            progress('Fresh candidate prefix/API validation',args.model+' '+str(case+1)+'/300 exact adversarial prefixes',phase=phase,counts_as_research=not args.audit)
            print('NEXT4 FRESH PREFIX',args.model,case+1,flush=True)
    atomic_json(out/'PREFIX_CASES.json',{'status':'PASS','cases':cases,'seed':20260908},immutable=True)
    data=prepare_dev(ROOT);features=None
    if args.model!='BL80':
        from src.next4.cache import build_design
        cid='B2_LOG_ENERGY_SEMITAIL' if args.model=='B2' else args.model
        features,_,_,_,_=build_design(data,read_json(BASE/'registrations'/(cid+'.json')),'full')
    datasets=Datasets(data,range(len(data.ids)));stream=entry.infer(datasets,directory)
    assert datasets.accessed==0 and next(stream) is None and datasets.accessed==0
    active=[False];writes=[];sys.addaudithook(write_audit(active,writes));fingerprint=hashlib.sha256();saved={};position=0
    for j,sid in enumerate(data.ids):
        h,o,_=data.get(int(sid));wanted=expected(h,o,None if features is None else features[position:position+len(o)])
        actual=np.empty(len(o),dtype=np.float32);active[0]=True
        try:
            for k in range(len(o)):
                value=next(stream);assert np.isscalar(value) and np.isfinite(value) and 0<=value<=1
                assert datasets.index==j and datasets.current.consumed==k+1;actual[k]=value
        finally:active[0]=False
        np.testing.assert_array_equal(actual,wanted);fingerprint.update(actual.tobytes());position+=len(o)
        if j<20:saved[j]=actual.copy()
        if (j+1)%250==0:
            progress('Fresh public API all-DEV equivalence',args.model+' '+str(j+1)+'/8000, strict current consumption and batch parity',phase=phase,counts_as_research=not args.audit)
            print('NEXT4 FRESH PUBLIC ALL DEV',args.model,j+1,flush=True)
    try:next(stream)
    except StopIteration:pass
    else:raise AssertionError('Extra all-DEV output')
    reordered=Datasets(data,list(reversed(saved)));repeat=entry.infer(reordered,directory);assert next(repeat) is None
    for j in reversed(saved):np.testing.assert_array_equal(np.asarray([next(repeat) for _ in saved[j]],dtype=np.float32),saved[j])
    try:next(repeat)
    except StopIteration:pass
    else:raise AssertionError('Extra reordered output')
    if args.model=='BL80':
        assert fingerprint.hexdigest()=='ff9589de5e233f3939647ea348694913261b3f25c74c7ab05930363a7386bc59'
    elif args.model=='B2':assert fingerprint.hexdigest()==read_json(BASE/'runtime_validation/B2_DIRECT/RESULTS.json')['output_fingerprint_sha256']
    elif args.audit:
        assert fingerprint.hexdigest()==read_json(BASE/'finalist_validation'/args.model/('API_'+args.model)/'RESULTS.json')['output_fingerprint_sha256']
    data.guard.check_state();assert not writes
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'model':args.model,'audit_phase':args.audit,
        'prefix_cases':300,'prefix_changed_suffix_and_shorter_horizon_exact':True,'offline_streaming_all_DEV_exact':True,
        'all_DEV_series':8000,'all_DEV_points':position,'output_fingerprint_sha256':fingerprint.hexdigest(),
        'first_none_before_data':True,'current_point_consumption_exact':True,'future_length_and_suffix_traps':True,
        'reordered_reset_series':20,'persistent_inference_write_attempts':len(writes),'new_fits':0,'seal_rows':0,
        'state_array_bytes_min':min(r['state_array_bytes'] for r in cases),'state_array_bytes_max':max(r['state_array_bytes'] for r in cases),
        'source_sha256':digest(Path(__file__)),'entry_sha256':digest(ROOT/(entry_name+'.py')),
        'files_sha256':{'PREFIX_CASES.json':digest(out/'PREFIX_CASES.json')}},immutable=True)
    print('NEXT4 FRESH PUBLIC API COMPLETE',args.model,flush=True)


if __name__=='__main__':main()
