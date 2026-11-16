"""All DEV through actual public infer, with one-shot streams and no filesystem writes."""
import os
import sys
sys.dont_write_bytecode=True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import argparse
import importlib
import hashlib
import json
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.data import prepare_dev
from src.next2.session import checkpoint
from src.utils.artifacts import write_json,sha256,utc_now


class Online:
    def __init__(self,points):
        self.points=points;self.consumed=0;self.iterated=False
    def __iter__(self):
        if self.iterated:raise AssertionError('Online iterator traversed twice')
        self.iterated=True;return self
    def __next__(self):
        if self.consumed==len(self.points):raise StopIteration
        value=self.points[self.consumed];self.consumed+=1;return value
    def __len__(self):raise AssertionError('Future online length inspected')
    def __length_hint__(self):raise AssertionError('Future online length hint inspected')
    def __array__(self,*args,**kwargs):raise AssertionError('Online suffix materialized')


class Datasets:
    def __init__(self,data,indices):
        self.data=data;self.indices=iter(indices);self.current=None;self.index=None;self.accessed=0
    def __iter__(self):return self
    def __next__(self):
        if self.current is not None:assert self.current.consumed==len(self.current.points)
        j=next(self.indices);self.index=j;self.accessed+=1
        ho,oo=self.data.series['historical_offsets'],self.data.series['online_offsets']
        self.current=Online(self.data.online[oo[j]:oo[j+1]])
        return self.data.historical[ho[j]:ho[j+1]],self.current
    def __len__(self):raise AssertionError('Dataset iterator length inspected')
    def __length_hint__(self):raise AssertionError('Dataset length hint inspected')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--role',choices=['DISTILLED','PERFORMANCE'],required=True);args=parser.parse_args()
    data=prepare_dev(ROOT)
    variant='DISTILLED_tree_used_v1' if args.role=='DISTILLED' else 'PERFORMANCE_compact_v1'
    artifact=ROOT/'artifacts/next2/canonical_deployment'/variant
    out=ROOT/'artifacts/next2/canonical_api_all_dev'/args.role;out.mkdir(parents=True,exist_ok=True)
    if args.role=='DISTILLED':
        source=ROOT/'artifacts/next2/runtime_pruning/J102_SERIAL_VARIANCE_512'
        receipt=json.loads((source/'RESULTS.json').read_text());path=source/'all_dev_predictions.npy'
        assert sha256(path)==receipt['files_sha256'][path.name]
        expected=hashlib.sha256(np.load(path,mmap_mode='r',allow_pickle=False).tobytes()).hexdigest()
        evidence=source/'RESULTS.json'
    else:
        evidence=ROOT/'artifacts/next2/entry_reproduction/PERFORMANCE/raw_entry_v1/SEMANTIC_REPRODUCTION.json'
        expected=json.loads(evidence.read_text())['all_dev_prediction_sha256']
    identity={'role':args.role,'script_sha256':sha256(Path(__file__)),'model_sha256':sha256(artifact/'model.sbnext2'),
        'descriptor_sha256':sha256(artifact/'model.json'),'expected_fingerprint':expected,
        'reference_evidence_sha256':sha256(evidence),'dev_manifest_sha256':sha256(data.directory/'MANIFEST.json')}
    if (out/'RESULTS.json').exists():
        r=json.loads((out/'RESULTS.json').read_text());assert r['status']=='PASS' and r['identity']==identity
        print('NEXT2 PUBLIC API RETAINED',args.role,flush=True);return
    write_json(out/'POLICY.json',identity)
    active=[True];attempts=[];null_opens=[]
    def audit(event,arguments):
        if not active[0]:return
        write=False
        if event=='open':
            path,mode,flags=arguments
            write=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if write and isinstance(path,(str,bytes,os.PathLike)) and os.path.normcase(os.path.normpath(os.fsdecode(path)))==os.path.normcase(os.path.normpath(os.devnull)):
                null_opens.append(event);return
        if event in ('os.mkdir','os.remove','os.rmdir','os.rename','os.chmod','os.truncate','os.link','os.symlink'):
            write=True
        if write:
            attempts.append(event);raise RuntimeError('Public inference attempted persistent filesystem write: '+event)
    sys.addaudithook(audit)
    tick=time.perf_counter();saved={};indices_by_id={int(sid):j for j,sid in enumerate(data.ids)}
    sample=json.loads((ROOT/'configs/next_latency_policy.json').read_text())['series_ids']
    try:
        entry=importlib.import_module('next2_canonical_'+args.role.lower()+'_main')
        source=Datasets(data,range(len(data.ids)));stream=entry.infer(source,artifact)
        assert source.accessed==0
        assert next(stream) is None and source.accessed==0
        digest=hashlib.sha256();count=0
        oo=data.series['online_offsets']
        for j,sid in enumerate(data.ids):
            values=np.empty(int(oo[j+1]-oo[j]),dtype=np.float32)
            for k in range(len(values)):
                value=next(stream)
                assert type(value) is float and np.isfinite(value) and 0<=value<=1
                assert source.index==j and source.current.consumed==k+1
                values[k]=value
            digest.update(values.tobytes());count+=len(values)
            if int(sid) in sample:saved[int(sid)]=values
            if (j+1)%1000==0:print('NEXT2 PUBLIC API EXACT',args.role,j+1,round(time.perf_counter()-tick,1),flush=True)
        try:next(stream)
        except StopIteration:pass
        else:raise AssertionError('Extra output after all observations')
        assert count==len(data.online) and digest.hexdigest()==expected
        reordered=Datasets(data,[indices_by_id[sid] for sid in reversed(sample)])
        stream=entry.infer(reordered,artifact);assert next(stream) is None and reordered.accessed==0
        for sid in reversed(sample):
            actual=np.asarray([next(stream) for _ in range(len(saved[sid]))],dtype=np.float32)
            np.testing.assert_array_equal(actual,saved[sid])
        try:next(stream)
        except StopIteration:pass
        else:raise AssertionError('Extra output after reordered series')
        assert not attempts
    finally:
        active[0]=False
    write_json(out/'RESULTS.json',{'status':'PASS','identity':identity,'created_utc':utc_now(),
        'series':len(data.ids),'points':count,'output_fingerprint_sha256':digest.hexdigest(),
        'first_none_before_any_dataset_access':True,'one_current_point_per_output':True,'no_length_or_suffix_access':True,
        'reverse_order_reset_series':len(sample),'persistent_filesystem_write_attempts':len(attempts),
        'null_device_opens':len(null_opens),'elapsed_seconds':time.perf_counter()-tick,'seal_rows':0,
        'scope':'Actual canonical public infer API and every DEV observation; fingerprint equivalence only, no in-sample quality score.'})
    checkpoint('Actual public canonical API all DEV '+args.role,['Isolated runtime profiling','Separate fresh-copy final reproduction'],
        artifacts=[(out/'RESULTS.json').relative_to(ROOT).as_posix()])
    print('NEXT2 PUBLIC API PASS',args.role,count,flush=True)


if __name__=='__main__':
    main()
