"""Freeze all target-free evidence per feature; no break label loader exists."""
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.next9
from src.next9.common import BASE, ROOT, atomic, read_json, digest, activity, utc_now
from src.next9.boundary import install_workspace_boundary, target_free_npz


def main(part):
    violations = install_workspace_boundary()
    lock = read_json(BASE / 'MOVEMENT_SCREEN_IMPLEMENTATION_LOCK_NEXT9.json')
    for name, sha in lock['source_sha256'].items():
        assert digest(ROOT / name) == sha, name
    with target_free_npz() as accesses:
        import numpy as np
        from src.next9.data import FeatureBank, selection_splits
        from src.next9.movement_screen import feature_inputs, auxiliary_targets, fit_ridge_bundle, evaluate
        from src.next9.states import directions
        from src.next9.io import save_npz
        bank = FeatureBank()
        with np.load(BASE / 'state_cache/INDEX.npz') as z:
            allrows = z['global_row']
        ids = bank.row_ids[allrows]
        groups = np.asarray([bank.guard.split.groups[int(s)] for s in ids], dtype=np.int32)
        times = bank.times[allrows]
        keep_common = np.isin(allrows, bank.forecast_rows(allrows,16))
        scopes = [(fold, inner, split, scope) for fold in range(5) for inner in [None,0,1,2]
                  for split, scope in enumerate(selection_splits(bank,fold,inner))]
        # Precompute membership masks once; no outcomes or final length is a predictor.
        plans = []
        for fold, inner, split, scope in scopes:
            for early in (False,True):
                train = np.flatnonzero(keep_common & np.isin(ids,list(scope.training_ids)) & ((times+16<=199) if early else True))
                valid = np.flatnonzero(keep_common & np.isin(ids,list(scope.validation_ids)) & ((times>=400) if early else True))
                scope.admit_training(ids[train]);scope.admit_validation(ids[valid])
                plans.append((fold,inner,split,scope,early,train,valid))
        feature_list = list(range(153))[part::12]
        for j in feature_list:
            directory = BASE / 'movement_screen' / f'F{j:03d}'
            if (directory/'DONE.json').exists():
                d=read_json(directory/'DONE.json')
                assert d['implementation_lock_sha256']==digest(BASE/'MOVEMENT_SCREEN_IMPLEMENTATION_LOCK_NEXT9.json')
                continue
            sd=read_json(BASE/'state_cache'/f'F{j:03d}'/'DONE.json')
            sp=BASE/'state_cache'/f'F{j:03d}'/'STATES.npz'
            assert digest(sp)==sd['states_sha256']
            with np.load(sp) as z:
                state={n:z[n] for n in z.files}
            x=feature_inputs(bank,allrows,j,state)
            current=np.asarray(bank.values[allrows,j],dtype=np.float32)
            epsilon=bank.series['epsilon'][bank.series_index[allrows],0,j]
            current_direction=directions(current-x[:,0],epsilon)
            y=np.zeros((len(allrows),15),dtype=np.float32)
            y[keep_common]=auxiliary_targets(bank,allrows[keep_common],j,state['S1_DIRECTION'][keep_common])
            receipts=[]
            for fold,inner,split,scope,early,tr,va in plans:
                name=f'outer_{fold}_'+('main' if inner is None else f'inner_{inner}')+f'_selection_{split}_'+('EARLY' if early else 'NORMAL')
                out=directory/name
                if (out/'DONE.json').exists():
                    done=read_json(out/'DONE.json')
                    for n,s in done['files_sha256'].items():assert digest(out/n)==s,n
                    receipts.append(dict(name=name,done_sha256=digest(out/'DONE.json')))
                    continue
                with activity(f'F{j:03d} target-free own forecast {name}','MOVEMENT_SCREEN','B5_SELF_DYNAMICS_VS_FILTER_MEMORY'):
                    out.mkdir(parents=True,exist_ok=True)
                    model=fit_ridge_bundle(x[tr],y[tr],ids[tr],groups[tr],scope)
                    records,gids,loss=evaluate(model,x[va],y[va],current[va],current_direction[va],groups[va],y[tr],current_direction[tr],groups[tr])
                    atomic(out/'MODEL.json',model.record())
                    atomic(out/'RESULTS.json',dict(status='COMPLETE',timestamp=utc_now(),feature=j,fold=fold,inner=inner,
                          selection_split=split,early=early,scope=scope.provenance(),records=records,
                          training_time_maximum=int(times[tr].max()),training_auxiliary_time_maximum=int((times[tr]+16).max()),
                          validation_time_minimum=int(times[va].min()),features_used=[j],
                          single_feature_id=j,three_horizons_all_retained=True,new_estimator_fit_calls=1,
                          no_break_label_or_tau_read=True,scope_excludes_all_future_outer_and_meta_validation_groups=True))
                    save_npz(out/'GROUP_LOSSES.npz',group_id=gids,loss=loss)
                    atomic(out/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),files_sha256={n:digest(out/n) for n in ['MODEL.json','RESULTS.json','GROUP_LOSSES.npz']}))
                    receipts.append(dict(name=name,done_sha256=digest(out/'DONE.json')))
            atomic(directory/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),feature=j,selection_panels=receipts,
                  estimator_fit_calls=len(plans),implementation_lock_sha256=digest(BASE/'MOVEMENT_SCREEN_IMPLEMENTATION_LOCK_NEXT9.json')))
            print('MOVEMENT_SCREEN_FEATURE_COMPLETE',part,j,len(plans),flush=True)
        assert not violations
        atomic(BASE/'movement_screen'/f'PART_{part}_DONE.json',dict(status='COMPLETE',timestamp=utc_now(),features=feature_list,
              no_break_label_or_tau_read=True,NPZ_members=sorted(set(accesses)),boundary_violations=violations))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--part',type=int,choices=range(12),required=True)
    main(p.parse_args().part)
