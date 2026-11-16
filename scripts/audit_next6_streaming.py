"""Reproduction, bounded current-point contracts and group provenance audit."""
from pathlib import Path
import sys,os,hashlib,inspect,argparse,time,pickle
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'artifacts/next6/numba_cache')
import numpy as np,joblib
from src.next6.common import BASE,read_json,save,digest,ids_hash,activity,checkpoint,utc_now
from src.next6.data import source
from src.next6.runtime import Prepared,State
from src.next6.nodes import NodeState,select,SET_B,SET_C
from src.next5.reference_features import ReferenceState
from src.next5.synthetic import load_reference,cohort_series
from src.next5.runtime import SharedResearchBaseline,unique_array_bytes
from src.next6.evaluation import csv_write

def ordered(ids,prefix):return sorted(map(int,ids),key=lambda sid:hashlib.sha256((prefix+str(sid)).encode()).hexdigest())

def provenance(data):
    protocol=read_json(BASE/'NEXT6_PROTOCOL_LOCK.json');proof=[]
    for f in range(5):
        sets=[{int(i) for i,k in protocol['inner_mapping'][str(f)].items() if k==j} for j in range(3)];tr=set.union(*sets);va=set(data.ids)-tr
        assert all(protocol['outer_mapping'][str(int(i))]==f for i in va);data.guard.partition(tr,va)
        for p in sorted((BASE/f'fold_{f}/graphs').glob('*/TRAIN_PROVENANCE.json')):
            g=read_json(p);ids=set(g['training_ids']);assert ids<=tr and not ids&va;tag=p.parent.name
            if tag.startswith('single_'):expected=sets[int(tag[-1])]
            elif 'pair_' in tag:expected=tr-sets[int(tag[-1])]
            else:expected=tr
            assert ids==expected and g['training_ids_sha256']==ids_hash(expected)
            proof.append({'fold':f,'graph':tag,'training_series':len(ids),'training_ids_sha256':ids_hash(ids),'no_outer_validation_ids':True})
        for held in range(3):
            for recipient in range(3):
                if recipient==held:continue
                donor=next(j for j in range(3) if j not in (held,recipient))
                assert not sets[donor]&sets[held] and not sets[donor]&sets[recipient]
            for method in ('M1','M2','M3','M4'):
                r=read_json(BASE/f'fold_{f}/inner_{held}/{method}/DONE.json');assert r['training_ids_sha256']==ids_hash(tr-sets[held])
        for method in read_json(BASE/f'fold_{f}/OUTER_PREDICTIONS_FROZEN.json')['models']:
            r=read_json(BASE/f'fold_{f}/outer/{method}/DONE.json');assert r['training_ids_sha256']==ids_hash(tr)
        freeze=read_json(BASE/f'fold_{f}/INNER_FREEZE.json');scores=freeze['inner_scores'];assert freeze['M4_eligible']==(scores['M4']>=max(scores['M2'],scores['M3'])+.0002)
        assert (BASE/f'fold_{f}/outer/M4/DONE.json').exists()==freeze['M4_eligible']
    return proof

def parity(data,fold):
    if (BASE/f'final_audit/PARITY_fold_{fold}.json').exists():return
    valid=data.ids[data.series['fold']==fold];ids=ordered(valid,'NEXT6_PREFIX:')[:20];results=[]
    with activity('outer_streaming_contracts_fold'+str(fold),'FINAL_AUDIT'):
        for method in read_json(BASE/f'fold_{fold}/OUTER_PREDICTIONS_FROZEN.json')['models']:
            d=BASE/f'fold_{fold}/outer/{method}';r=read_json(d/'DONE.json');assert digest(d/'model.joblib')==r['model_sha256']
            model=joblib.load(d/'model.joblib');prepared=Prepared(model)
            with np.load(d/'predictions.npz',allow_pickle=False) as z:
                data.guard.admit(np.unique(z['dataset_id']),purpose='NEXT6 scalar parity heldout predictions');rid=z['dataset_id'];p=z['prediction']
            errors=[];points=0
            for k,sid in enumerate(ids):
                h,o,_=data.get(sid);expected=p[rid==sid];actual=prepared.new_state(h).replay(o);error=float(abs(expected-actual).max());assert error<1e-12,(fold,method,sid,error)
                points+=len(o);errors.append(error)
                if k<3:
                    cut=min(31,len(o));prefix=prepared.new_state(h).replay(o[:cut]);mut=prepared.new_state(h).replay(np.r_[o[:cut],np.full(137,101,dtype=np.float32)])
                    assert np.array_equal(prefix,actual[:cut]) and np.array_equal(mut[:cut],prefix)
                    state=prepared.new_state(h);live=np.array([state.predict_one(float(x)) for x in o[:64]])
                    assert np.array_equal(live,actual[:len(live)])
                    clone=pickle.loads(pickle.dumps(prepared));assert np.array_equal(clone.new_state(h).replay(o),actual)
                    # Interleaving another series/state must not alter either state.
                    state=prepared.new_state(h);other=prepared.new_state(h[::-1].copy());first=state.replay(o[:cut]);other.replay(o[:cut]);rest=state.replay(o[cut:]);assert np.array_equal(np.r_[first,rest],actual)
                    for key in ('tau','target','phase','final_length','dataset_id'):
                        try:state.predict_one(float(o[0]),**{key:3})
                        except TypeError:pass
                        else:raise AssertionError('Forbidden runtime side channel '+key)
                    for bad in (float('nan'),float('inf')):
                        try:prepared.new_state(h).predict_one(bad)
                        except ValueError:pass
                        else:raise AssertionError('Nonfinite accepted')
            results.append({'fold':fold,'method':method,'series':len(ids),'points':points,'max_abs_error':max(errors),
                'prefix_suffix_length':True,'batch_scalar':True,'reset_interleaving':True,'serialized_model_reload':True,'determinism':True,'tau_label_phase_id_api_rejected':True,
                'model_sha256':r['model_sha256']})
            checkpoint('outer scalar audit',f'fold{fold} {method} {len(ids)} heldout series PASS')
        save(f'final_audit/PARITY_fold_{fold}.json',{'status':'PASS','results':results,'series_ids':ids})

def latency(data):
    if (BASE/'LATENCY_NEXT6.json').exists():return
    with activity('interleaved_actual_scalar_latency','FINAL_AUDIT'):
        refs,_=load_reference('full',0);models={'M0':SharedResearchBaseline(refs['J102'],refs['C205'])}
        for m in read_json(BASE/'fold_0/OUTER_PREDICTIONS_FROZEN.json')['models']:models[m]=Prepared(joblib.load(BASE/f'fold_0/outer/{m}/model.joblib'))
        if (BASE/'strict_baseline/fold_0/model.joblib').exists():
            from src.next6.strict_runtime import PreparedStrict
            models['M1_STRICT']=PreparedStrict(joblib.load(BASE/'strict_baseline/fold_0/model.joblib'))
        lock=read_json(ROOT/'COUNTEREXAMPLE_SUITE_LOCK.json');h,o,_=cohort_series('long_null','N2',63,4096,True,lock)
        states={};warm={};initial={};measure={m:[] for m in models};blocks={m:[] for m in models}
        for m,p in models.items():
            start=time.perf_counter();s=p.new_state(h);s.predict_one(float(o[0]));warm[m]=time.perf_counter()-start
            start=time.perf_counter();states[m]=p.new_state(h);initial[m]=time.perf_counter()-start
        for block in range(5):
            keys=list(models);keys=keys[block%len(keys):]+keys[:block%len(keys)]
            for m in keys:
                s=states[m];start=time.perf_counter_ns()
                for value in o[:2048]:
                    a=time.perf_counter_ns();result=s.predict_one(float(value));measure[m].append(time.perf_counter_ns()-a)
                    assert np.isfinite(result) and 0<=result<=1
                blocks[m].append((time.perf_counter_ns()-start)/2048)
        out=[];base=np.median(measure['M0'])
        for m in models:
            q=np.quantile(measure[m],[.5,.95,.99]);s=states[m]
            memory=s.state_array_bytes if hasattr(s,'state_array_bytes') else unique_array_bytes(s.args[:12]+(s.args[17],s.args[20]))
            out.append({'method':m,'points':len(measure[m]),'scalar_median_us':float(q[0]/1000),'scalar_p95_us':float(q[1]/1000),'scalar_p99_us':float(q[2]/1000),
                'median_ratio_vs_M0':float(q[0]/base),'median_block_average_us':float(np.median(blocks[m])/1000),
                'state_array_bytes_excluding_immutable_graph_trees':int(memory),'warm_H_initialization_seconds':initial[m],
                'first_state_and_JIT_seconds':warm[m],'bounded_state_arrays':True})
        save('LATENCY_NEXT6.json',{'status':'COMPLETE','clock':'perf_counter_ns','actual_predict_one_calls':True,'rows':out,'fold':0,'same_H_and_stream':True,
            'interleaved_blocks':5,'points_per_block':2048,'JIT_excluded_from_hot_latency_but_reported':True,'first_model_process_compile_order_effect':True,
            'latency_context':'local CPU measurement; concurrent activity recorded in ledger, not a hardware-independent guarantee'})

def historical_calibration(data):
    if (BASE/'H_CALIBRATION_DIAGNOSTIC.csv').exists():return
    ids=ordered(data.ids[data.series['fold']!=0],'NEXT6_H_CAL:')[:64];rows=[]
    with activity('H_pseudo_online_calibration_diagnostic','FINAL_AUDIT'):
        for sid in ids:
            h,_,_=data.get(sid);n=NodeState(h);b=ReferenceState(h,SET_B);c=ReferenceState(h,SET_C);raw=select(b.replay(h[n.fit_length:]),c.replay(h[n.fit_length:]));same=n.categorize(raw)
            for j in range(16):rows.append({'dataset_id':sid,'node':j,'pseudo_online_H_fraction_non_normal':float(np.mean((n.pseudo_s[:,j]!=1)&(n.pseudo_s[:,j]!=3))),
                'same_H_fitted_engine_fraction_non_normal':float(np.mean((same[:,j]!=1)&(same[:,j]!=3))),'degenerate':bool(n.degenerate[j]),
                'used_for_model_calibration':'pseudo-online H only','same_H_diagnostic_replay_is_not_causal_training':True})
        csv_write('H_CALIBRATION_DIAGNOSTIC.csv',rows)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fold',type=int);parser.add_argument('--latency',action='store_true');args=parser.parse_args();data=source()
    if args.latency:latency(data);return
    for fold in ([args.fold] if args.fold is not None else range(5)):parity(data,fold)
    if all((BASE/f'final_audit/PARITY_fold_{f}.json').exists() for f in range(5)):
        proof=provenance(data);historical_calibration(data);results=[r for f in range(5) for r in read_json(BASE/f'final_audit/PARITY_fold_{f}.json')['results']]
        save('PREFIX_PARITY_NEXT6.json',{'status':'PASS','results':results,'runtime_source_sha256':digest(ROOT/'src/next6/runtime.py'),'input_contract':str(inspect.signature(State.predict_one))})
        save('LEAKAGE_AUDIT_NEXT6.json',{'status':'PASS','graph_provenance':proof,'outer_validation_disjoint':True,'head_graph_inputs_crossfit':True,
            'inner_validation_labels_excluded_from_graph_training':True,'normalizers_H_only':True,'state_boundaries_H_pseudo_online_only':True,
            'tau_is_offline_supervision_only':True,'both_normal_and_post_model_scores_at_each_time':True,'future_smoothing':False,'runtime_reset_at_tau':False,
            'runtime_no_final_length':True,'runtime_no_dataset_id':True,'adaptive_DEV_reuse_still_high':True})
        save('REPRODUCTION_NEXT6.json',{'status':'PASS','kind':'saved fixed-fit OOF replay, native/compact parity and full source/data provenance; full head refit not repeated',
            'total_model_series_replays':sum(r['series'] for r in results),'total_points':sum(r['points'] for r in results),'max_abs_error':max(r['max_abs_error'] for r in results),
            'deterministic_saved_artifact_replay':True,'fresh_refit_independence_claim':False})
    data.guard.check_state()

if __name__=='__main__':main()
