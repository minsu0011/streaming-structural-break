"""Actual read-only API, all ResearchPool OOF points, prefixes and resets."""
import os
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import argparse
import hashlib
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.next5.artifacts import BASE,atomic_json,read_json,digest,utc_now,progress
from src.next5.data import prepare_research,admitted_npz
from src.next5.stream_contracts import Datasets
from src.scoring.ts_auc import ts_auc


def collect(infer,descriptor,historical,online):
    source=Datasets([(0,historical,online)]);stream=infer(source,descriptor)
    assert source.accessed==0 and next(stream) is None and source.accessed==0
    output=np.empty(len(online),dtype=np.float64)
    for j in range(len(online)):
        output[j]=next(stream);assert source.accessed==1 and source.current.consumed==j+1
    try:next(stream)
    except StopIteration:pass
    else:raise AssertionError('Extra output after current observations')
    return output


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',required=True);args=parser.parse_args();cid=args.candidate
    policy=read_json(BASE/'runtime_validation/API_POLICY_LOCK.json');assert cid in policy['candidates']
    for name,sha in policy['sources_sha256'].items():assert digest(ROOT/name)==sha
    out=BASE/'runtime_validation/api'/cid;out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'START.json',{'created_utc':utc_now(),'candidate':cid,'policy_sha256':digest(BASE/'runtime_validation/API_POLICY_LOCK.json')},immutable=True)
    data=prepare_research();metadata=data.rows();ids=data.ids;offsets=data.series['online_offsets'];active=[False];attempts=[];null_opens=[]
    def audit(event,arguments):
        if not active[0]:return
        write=False
        if event=='open':
            path,mode,flags=arguments
            write=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if write and isinstance(path,(str,bytes,os.PathLike)) and os.path.normcase(os.path.normpath(os.fsdecode(path)))==os.path.normcase(os.path.normpath(os.devnull)):
                null_opens.append(event);return
        if event in ('os.mkdir','os.remove','os.rmdir','os.rename','os.chmod','os.truncate','os.link','os.symlink'):write=True
        if write:attempts.append(event);raise RuntimeError('Research inference attempted persistent filesystem write: '+event)
    sys.addaudithook(audit);started=time.perf_counter();folds=[];all_max=0.;all_exact=True;real_points=0;reset_series=0
    try:
        active[0]=True
        from src.next5.streaming import infer,load_prepared
        descriptor=ROOT/policy['candidates'][cid]['descriptors']['0'];rng=np.random.default_rng(policy['prefix_seed'])
        for case in range(policy['prefix_cases']):
            h=(rng.standard_t(5,[192,256,512,1024][case%4])*(.5+case%3)).astype(np.float32)
            o=rng.normal(size=160).astype(np.float32);cutoff=case%150+1
            longer=np.r_[o,rng.normal(size=73).astype(np.float32)];longer[cutoff:]=rng.normal(0,20,len(longer)-cutoff).astype(np.float32)
            whole=collect(infer,descriptor,h,o);changed=collect(infer,descriptor,h,longer);short=collect(infer,descriptor,h,o[:cutoff])
            np.testing.assert_array_equal(whole[:cutoff],changed[:cutoff]);np.testing.assert_array_equal(whole[:cutoff],short)
            if case%50==49:print('NEXT5 API PREFIX',cid,case+1,flush=True)
        for fold in range(5):
            active[0]=False;descriptor=ROOT/policy['candidates'][cid]['descriptors'][str(fold)]
            source_dir=BASE/'experiments'/cid/'full'/('fold_'+str(fold));receipt=read_json(source_dir/'RESULTS.json')
            path=source_dir/'predictions.npz';assert digest(path)==receipt['predictions_file_sha256']
            expected=admitted_npz(path,data.guard,'Actual causal API held-out prediction parity')
            indices=[j for j,sid in enumerate(ids) if data.guard.split.assignment(int(sid))[1]==fold]
            def records(order):
                for j in order:
                    h,o,_=data.get(int(ids[j]));yield int(ids[j]),h,o
            stream_source=Datasets(records(indices));active[0]=True;stream=infer(stream_source,descriptor)
            assert stream_source.accessed==0 and next(stream) is None and stream_source.accessed==0
            actual=np.empty(len(expected['prediction']),dtype=np.float64);cursor=0;saved={}
            for number,j in enumerate(indices):
                n=int(offsets[j+1]-offsets[j]);values=np.empty(n,dtype=np.float64)
                for k in range(n):
                    values[k]=next(stream);assert stream_source.sid==int(ids[j]) and stream_source.current.consumed==k+1
                np.testing.assert_array_equal(expected['dataset_id'][cursor:cursor+n],np.full(n,ids[j]))
                actual[cursor:cursor+n]=values;cursor+=n
                if number<20:saved[j]=values.copy()
                if number%256==255:print('NEXT5 REAL API',cid,fold,number+1,cursor,flush=True)
            try:next(stream)
            except StopIteration:pass
            else:raise AssertionError('Extra output after held-out fold')
            assert cursor==len(actual);difference=float(np.max(np.abs(actual-expected['prediction'])))
            if difference>1e-15:raise AssertionError('Scalar and frozen offline probability mismatch')
            reorder=Datasets(records(reversed(list(saved))));stream=infer(reorder,descriptor)
            assert next(stream) is None and reorder.accessed==0
            for j in reversed(list(saved)):
                values=np.asarray([next(stream) for _ in range(len(saved[j]))]);np.testing.assert_array_equal(values,saved[j]);reset_series+=1
            try:next(stream)
            except StopIteration:pass
            else:raise AssertionError('Extra reordered stream output')
            active[0]=False;rows=expected['global_row']
            score=float(ts_auc(metadata['target'][rows],actual,metadata['time_online'][rows]))
            assert abs(score-receipt['TS_AUC'])<1e-14
            record={'fold':fold,'series':len(indices),'points':len(actual),'max_abs_probability_difference':difference,
                'bitwise_float64_equal':bool(np.array_equal(actual,expected['prediction'])),'actual_API_TS_AUC':score,'frozen_offline_TS_AUC':receipt['TS_AUC'],
                'actual_output_fingerprint_sha256':hashlib.sha256(actual.tobytes()).hexdigest(),'descriptor_sha256':digest(descriptor)}
            atomic_json(out/('fold_'+str(fold)+'.json'),record,immutable=True);folds.append(record)
            real_points+=len(actual);all_max=max(all_max,difference);all_exact=all_exact and record['bitwise_float64_equal']
            progress('Actual fitted-fold streaming parity complete',cid+' fold '+str(fold)+'; no future access, exact metric parity')
        assert not attempts
    finally:active[0]=False
    data.guard.check_state();atomic_json(out/'RESULTS.json',{'created_utc':utc_now(),'status':'PASS','candidate':cid,'series':len(ids),'points':real_points,
        'prefix_cases':policy['prefix_cases'],'prefix_suffix_and_final_length_invariance':True,'first_none_before_dataset_access':True,
        'exactly_one_current_observation_per_output':True,'reverse_order_reset_series':reset_series,'persistent_filesystem_write_attempts':len(attempts),
        'null_device_opens':len(null_opens),'max_abs_probability_difference':all_max,'bitwise_float64_equal':all_exact,
        'official_TS_AUC_parity':True,'fold_results':folds,'new_fits':0,'shadow_rows':0,'seal_rows':0,
        'elapsed_seconds':time.perf_counter()-started,'scope':'Actual API across each of five held-out fitted artifacts; no historical all-DEV or Shadow replay.'},immutable=True)
    progress('ResearchPool public-stream contract verified',cid+' all five fitted artifacts and all ResearchPool OOF points, 300 future-prefix traps and reversed resets')


if __name__=='__main__':main()
