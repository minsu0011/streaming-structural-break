"""Audit first-readiness and one-pass NEXT inference, including Python audit IO."""
import os
import sys
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import argparse
import json
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.next.inference import infer,implementation_hash
from src.next.data import prepare_dev
from src.next.runtime_benchmark import load_benchmark_model
from src.utils.artifacts import write_json,sha256,utc_now


class GuardedOnline:
    def __init__(self,values):
        self.iterator = iter(values)
        self.started,self.pending,self.count = False,False,0
    def __iter__(self):
        if self.started:
            raise AssertionError('Online iterator was replayed')
        self.started = True
        return self
    def __len__(self):
        raise AssertionError('Future online length queried')
    def __getitem__(self,key):
        raise AssertionError('Future online values indexed')
    def __array__(self,*args,**kwargs):
        raise AssertionError('Online iterable materialized')
    def __next__(self):
        if self.pending:
            raise AssertionError('Read-ahead before returning the current prediction')
        value = next(self.iterator)
        self.pending,self.count = True,self.count+1
        return value
    def acknowledge(self):
        if not self.pending:
            raise AssertionError('Output without an input')
        self.pending = False


class GuardedDatasets:
    def __init__(self,pairs):
        self.pairs,self.started,self.count = iter(pairs),False,0
    def __iter__(self):
        if self.started:
            raise AssertionError('Outer iterator was replayed')
        self.started = True
        return self
    def __len__(self):
        raise AssertionError('Outer dataset count queried')
    def __getitem__(self,key):
        raise AssertionError('Outer dataset indexed')
    def __next__(self):
        value = next(self.pairs)
        self.count += 1
        return value


def run_guarded(examples,directory,*,prefix_only=False):
    streams = [GuardedOnline(points) for h,points,n in examples]
    outer = GuardedDatasets([(h,stream) for (h,points,n),stream in zip(examples,streams)])
    generator = infer(outer,directory)
    if next(generator) is not None or outer.count!=0 or any(stream.count for stream in streams):
        raise AssertionError('Readiness consumed an input')
    predictions = []
    for (h,points,n),stream in zip(examples,streams):
        values = []
        for _ in range(n):
            value = next(generator)
            if type(value) is not float or not np.isfinite(value) or not 0.<=value<=1.:
                raise AssertionError('Invalid scalar output')
            values.append(value)
            stream.acknowledge()
        if stream.count!=n:
            raise AssertionError('Input/output counts differ')
        predictions.append(np.asarray(values,dtype=np.float32))
    if prefix_only:
        generator.close()
    else:
        try:
            next(generator)
        except StopIteration:
            pass
        else:
            raise AssertionError('More outputs than online observations')
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate',default='V007_VARIANCE_FORECAST')
    parser.add_argument('--tag',default='v2')
    args = parser.parse_args()
    directory = ROOT/'artifacts/next/deployment/contracts'/args.candidate/args.tag
    descriptor = json.loads((directory/'model.json').read_text(encoding='utf-8'))
    if descriptor['inference_sha256']!=implementation_hash():
        raise RuntimeError('Requested fixture has a different inference source')
    data = prepare_dev(ROOT)
    _,model_path,expected = load_benchmark_model(ROOT,data,args.candidate)
    policy = json.loads((ROOT/'configs/next_latency_policy.json').read_text(encoding='utf-8'))
    ids = policy['series_ids'][:32]
    examples = []
    for sid in ids:
        h,o,_ = data.get(sid)
        examples.append((h,o,len(o)))
    enabled = [False]
    observed_events = [0]
    mutation_events = {'os.remove','os.rename','os.mkdir','os.rmdir','os.link','os.symlink','os.chmod','os.truncate'}
    def audit(event,arguments):
        if not enabled[0]:
            return
        if event=='open':
            path,mode,flags = arguments
            writes = isinstance(mode,str) and any(char in mode for char in 'wax+')
            writes = writes or (isinstance(flags,int) and bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
            if writes:
                raise AssertionError(f'Inference attempted file write: {path}')
            observed_events[0] += 1
        elif event in mutation_events:
            raise AssertionError('Inference attempted filesystem mutation: '+event)
    sys.addaudithook(audit)
    tick = time.perf_counter()
    enabled[0] = True
    try:
        # No inference warm-up before this guard: first readiness compilation,
        # model reload and each new H are included. Module import is excluded.
        first = run_guarded(examples,directory)
        repeated = run_guarded(examples,directory)
        for sid,left,right in zip(ids,first,repeated):
            np.testing.assert_array_equal(left,right)
            np.testing.assert_array_equal(left,expected[sid])
        suffix_cases = 0
        for (h,o,n),reference in zip(examples,first):
            for prefix in sorted(set(min(length,n) for length in (1,8,64,257))):
                for suffix in (np.full(5,-1e5,dtype=np.float32),np.full(17,np.nan,dtype=np.float32),np.linspace(-500,500,257,dtype=np.float32)):
                    altered = np.r_[o[:prefix],suffix]
                    actual = run_guarded([(h,altered,prefix)],directory,prefix_only=True)[0]
                    np.testing.assert_array_equal(actual,reference[:prefix])
                    suffix_cases += 1
        # An observed H with an empty online iterator still has no prediction.
        assert len(run_guarded([(examples[0][0],np.empty(0,dtype=np.float32),0)],directory)[0])==0
    finally:
        enabled[0] = False
    report = {'status':'PASS','candidate':args.candidate,'fixture_tag':args.tag,'created_utc':utc_now(),
        'real_dev_series':len(examples),'original_points':sum(n for h,o,n in examples),'modified_future_suffix_cases':suffix_cases,
        'first_readiness_included_in_write_guard':True,'module_import_before_generator_excluded':True,
        'python_audit_read_open_events':observed_events[0],'file_writes_or_mutations':0,
        'repeat_and_oof_predictions_bitwise_equal':True,'no_read_ahead_or_online_length_or_materialization':True,
        'outer_and_inner_iterators_single_pass':True,'readiness_consumes_zero_outer_or_online_items':True,
        'empty_online_sequence_emits_no_prediction':True,'future_nan_suffix_never_consumed':True,
        'model_sha256':sha256(model_path),'descriptor_sha256':sha256(directory/'model.json'),
        'inference_sha256':implementation_hash(),'script_sha256':sha256(Path(__file__)),
        'elapsed_seconds':time.perf_counter()-tick,'seal_rows':0,'reduced_new_usage':0,
        'scope':'Local generator/IO/causality contract, including cold inference readiness after module imports. This is not an authenticated SDK test or cloud filesystem proof.'}
    out = ROOT/'artifacts/next/inference_contract'/args.candidate
    write_json(out/(args.tag+'.json'),report)
    print('NEXT INFERENCE CONTRACT PASS',args.candidate,report['original_points'],suffix_cases,flush=True)


if __name__=='__main__':
    main()
