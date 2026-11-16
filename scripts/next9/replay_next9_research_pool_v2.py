"""Actual raw RP stream replay against frozen full-point M predictions."""
from pathlib import Path
import sys
import argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now
from src.next9.boundary import install_workspace_boundary,target_free_npz


def main(fold,mode,part):
    violations=install_workspace_boundary()
    lock=read_json(BASE/'LIVE_REPLAY_IMPLEMENTATION_LOCK_V2_NEXT9.json')
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    global_path=BASE/'GLOBAL_PREDICTIONS_FROZEN_NEXT9.json';global_freeze=read_json(global_path)
    assert global_freeze['status']=='FROZEN'
    with target_free_npz() as accesses:
        import numpy as np
        import pickle
        import hashlib
        import copy
        from src.next9.data import FeatureBank,outer_scope
        from src.next9.runtime_loading_v2 import load_heads,load_baseline
        from src.next9.streaming import ResearchStream
        from src.next9.io import save_npz
        from src.next5.runtime import ScalarDetector
        bank=FeatureBank();scope=outer_scope(bank,fold)
        root=BASE/'live_replay'/scope.name/mode/f'part_{part}'
        if (root/'DONE.json').exists():return
        member=next(r for r in global_freeze['records'] if r['fold']==fold and r['mode']==mode)
        frozen_path=BASE/member['path'];assert digest(frozen_path)==member['sha256'];frozen=read_json(frozen_path)
        assert digest(frozen_path.parent/'PREDICTIONS.npz')==frozen['prediction_sha256']
        with np.load(frozen_path.parent/'PREDICTIONS.npz') as z:
            expected_rows=z['global_row'];expected={k:z[k] for k in frozen['methods']}
        heads=load_heads(fold,mode);baseline=load_baseline(fold,mode)
        static_before=hashlib.sha256(pickle.dumps(heads,protocol=5)).hexdigest()
        preflight=read_json(BASE/'PREFLIGHT_NEXT9.json');directory=ROOT/preflight['RP_cache'];manifest=read_json(directory/'MANIFEST.json')
        for name in ['historical.npy','online.npy']:assert digest(directory/name)==manifest['files_sha256'][name]
        historical=np.load(directory/'historical.npy',mmap_mode='r');online=np.load(directory/'online.npy',mmap_mode='r')
        indices=np.flatnonzero(bank.folds==fold)[part::2]
        if mode=='EARLY':indices=indices[::-1]
        records=[];all_rows=[];all_predictions={k:[] for k in expected}
        for s in indices:
            sid=int(bank.ids[s]);out=root/f'S{sid}'
            if (out/'DONE.json').exists():
                receipt=read_json(out/'DONE.json');assert digest(out/'PREDICTIONS.npz')==receipt['prediction_sha256']
                with np.load(out/'PREDICTIONS.npz') as z:
                    all_rows.append(z['global_row'])
                    for name in expected:all_predictions[name].append(z[name])
                records.append(receipt);continue
            with activity(f'{scope.name} {mode} original RP series{sid} scalar M replay and state restore','FULL_RP_STREAM_REPLAY','B9_NORMALIZATION_STATE_AVAILABILITY'):
                lo,hi=map(int,bank.offsets[s:s+2]);hlo,hhi=map(int,bank.historical_offsets[s:s+2])
                h=np.asarray(historical[hlo:hhi]);o=np.asarray(online[lo:hi]);scope.admit_validation([sid])
                stream=ResearchStream(heads,h);base=baseline.new_state(h)
                np.testing.assert_array_equal(stream.features.history,bank.series['seed_values'][s])
                np.testing.assert_array_equal(stream.features.context,bank.series['context'][s])
                np.testing.assert_array_equal(stream.features.epsilon.astype(np.float32),bank.series['epsilon'][s])
                output={k:[] for k in expected};state_restore=False;first=[]
                for t,point in enumerate(o):
                    b=base.predict_one(float(point));prediction=stream.predict_one(float(point),b)
                    np.testing.assert_array_equal(stream.bank.past[-1],bank.values[lo+t])
                    for name in output:output[name].append(prediction[name])
                    if t<17:first.append(dict(prediction))
                    if t==min(73,len(o)-1):
                        # Only mutable streaming state is serialized; the frozen
                        # fitted model object is independently hashed below.
                        fstate,pbank,args=pickle.loads(pickle.dumps((stream.features,stream.bank,copy.deepcopy(base.args)),protocol=5))
                        stream.features=fstate;stream.bank=pbank;base=ScalarDetector(base.kernel,args,base.prepared);state_restore=True
                output={k:np.asarray(v,dtype=np.float64) for k,v in output.items()}
                rows=np.arange(lo,hi,dtype=np.int64);keep=bank.times[rows]>=400 if mode=='EARLY' else np.ones(len(rows),dtype=bool)
                loc=np.searchsorted(expected_rows,rows[keep]);np.testing.assert_array_equal(expected_rows[loc],rows[keep])
                errors={}
                for name,prediction in output.items():
                    assert np.isfinite(prediction).all() and np.all((prediction>=0)&(prediction<=1)),name
                    errors[name]=float(np.max(np.abs(prediction[keep]-expected[name][loc]))) if keep.any() else 0.
                    assert errors[name]<=1e-7,(sid,mode,name,errors[name])
                if heads.correction is None:np.testing.assert_array_equal(output['M7'],output['M0'])
                # Fresh series reset and changed, longer suffix. Only the same
                # observed prefix is consumed; no final length enters the API.
                again=ResearchStream(heads,h);again_base=baseline.new_state(h)
                changed=np.r_[o[:17],-7.*o[17:]+.375,np.array([999.,-999.],dtype=np.float32)]
                for t,point in enumerate(changed[:min(17,len(o))]):
                    actual=again.predict_one(float(point),again_base.predict_one(float(point)))
                    for name in first[t]:assert np.float64(actual[name]).tobytes()==np.float64(first[t][name]).tobytes(),(sid,name,t)
                save_npz(out/'PREDICTIONS.npz',global_row=rows[keep],**{k:v[keep] for k,v in output.items()})
                receipt=dict(status='COMPLETE',timestamp=utc_now(),dataset_id=sid,fold=fold,mode=mode,points=len(o),evaluated_points=int(keep.sum()),
                    maximum_absolute_errors=errors,feature_scalar_cache_bitwise=True,reset_prefix17_bitwise=True,
                    save_restore_mutable_state_at74=state_restore,altered_extended_future_suffix_not_exposed=True,
                    probability_tolerance=1e-7,TS_AUC_parity_separate_pending=True,M7_alias_exact_BL=heads.correction is None,
                    only_H_and_current_scalar_exposed=True,observed_feature_memory_vectors=17,
                    prediction_sha256=digest(out/'PREDICTIONS.npz'),source_prediction_freeze_sha256=digest(frozen_path),break_labels_or_tau_used=False)
                atomic(out/'DONE.json',receipt);records.append(receipt);all_rows.append(rows[keep])
                for name in expected:all_predictions[name].append(output[name][keep])
        static_after=hashlib.sha256(pickle.dumps(heads,protocol=5)).hexdigest()
        assert static_before==static_after,'Fitted classifier/transition parameters mutated during inference'
        rows=np.concatenate(all_rows);order=np.argsort(rows)
        save_npz(root/'PREDICTIONS.npz',global_row=rows[order],**{k:np.concatenate(v)[order] for k,v in all_predictions.items()})
        assert not violations
        atomic(root/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),scope=scope.provenance(),fold=fold,mode=mode,part=part,
            original_series=len(indices),points=sum(r['points'] for r in records),evaluated_points=len(rows),
            series_ids=[r['dataset_id'] for r in records],series_order='REVERSED' if mode=='EARLY' else 'FORWARD',
            max_errors={k:max(r['maximum_absolute_errors'][k] for r in records) for k in expected},
            no_fitted_parameter_updates=True,model_pickle_sha256=static_after,break_labels_or_tau_used=False,
            prediction_sha256=digest(root/'PREDICTIONS.npz'),global_freeze_sha256=digest(global_path),NPZ_members=sorted(set(accesses))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',required=True,type=int,choices=range(5));p.add_argument('--mode',required=True,choices=['NORMAL','EARLY']);p.add_argument('--part',required=True,type=int,choices=[0,1]);a=p.parse_args();main(a.fold,a.mode,a.part)
