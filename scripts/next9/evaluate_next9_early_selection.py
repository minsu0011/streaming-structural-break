"""Strict early candidate selection validation; existing early fits only."""
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.next9
from src.next9.common import BASE, ROOT, atomic, read_json, digest, activity, utc_now
from src.next9.boundary import install_workspace_boundary, target_free_npz


def main(part):
    violations=install_workspace_boundary()
    lock=read_json(BASE/'EARLY_SELECTION_IMPLEMENTATION_LOCK_NEXT9.json')
    for name,sha in lock['source_sha256'].items():
        assert digest(ROOT/name)==sha,name
    with target_free_npz() as accesses:
        import numpy as np
        from src.next9.data import FeatureBank,selection_splits
        from src.next9.movement_screen import RidgeBundle,feature_inputs,auxiliary_targets,evaluate
        from src.next9.states import directions
        from src.next9.io import save_npz
        bank=FeatureBank()
        with np.load(BASE/'state_cache/INDEX.npz') as z:allrows=z['global_row']
        ids=bank.row_ids[allrows]
        groups=np.asarray([bank.guard.split.groups[int(s)] for s in ids],dtype=np.int32)
        times=bank.times[allrows]
        keep=np.isin(allrows,bank.forecast_rows(allrows,16)) & (times+16<=199)
        plans=[]
        for fold in range(5):
            for inner in [None,0,1,2]:
                for split,scope in enumerate(selection_splits(bank,fold,inner)):
                    tr=np.flatnonzero(keep & np.isin(ids,list(scope.training_ids)))
                    va=np.flatnonzero(keep & np.isin(ids,list(scope.validation_ids)))
                    scope.admit_training(ids[tr]);scope.admit_validation(ids[va])
                    plans.append((fold,inner,split,scope,tr,va))
        features=list(range(153))[part::4]
        for j in features:
            output=BASE/'early_selection'/f'F{j:03d}'
            if (output/'DONE.json').exists():
                assert read_json(output/'DONE.json')['implementation_lock_sha256']==digest(BASE/'EARLY_SELECTION_IMPLEMENTATION_LOCK_NEXT9.json')
                continue
            with activity(f'F{j:03d} early<=199 excluded-group eligibility validation; no new model fit','EARLY_SELECTION','B4_EARLY_LATE_GENERALIZATION'):
                sp=BASE/'state_cache'/f'F{j:03d}'/'STATES.npz'
                assert digest(sp)==read_json(sp.parent/'DONE.json')['states_sha256']
                with np.load(sp) as z:state={n:z[n] for n in z.files}
                x=feature_inputs(bank,allrows,j,state)
                current=np.asarray(bank.values[allrows,j],dtype=np.float32)
                epsilon=bank.series['epsilon'][bank.series_index[allrows],0,j]
                current_direction=directions(current-x[:,0],epsilon)
                y=np.zeros((len(allrows),15),dtype=np.float32)
                y[keep]=auxiliary_targets(bank,allrows[keep],j,state['S1_DIRECTION'][keep])
                receipts=[]
                for fold,inner,split,scope,tr,va in plans:
                    name=f'outer_{fold}_'+('main' if inner is None else f'inner_{inner}')+f'_selection_{split}_EARLY'
                    original=BASE/'movement_screen'/f'F{j:03d}'/name
                    done=read_json(original/'DONE.json')
                    assert digest(original/'MODEL.json')==done['files_sha256']['MODEL.json']
                    original_result=read_json(original/'RESULTS.json')
                    assert original_result['training_auxiliary_time_maximum']<=199
                    record=read_json(original/'MODEL.json')
                    assert record['training']['scope']==scope.name
                    model=RidgeBundle(*(np.asarray(record[k],dtype=float) for k in ['center','scale','coefficients','intercept']),record['training'])
                    result,gids,loss=evaluate(model,x[va],y[va],current[va],current_direction[va],groups[va],y[tr],current_direction[tr],groups[tr])
                    out=output/name
                    out.mkdir(parents=True,exist_ok=True)
                    atomic(out/'RESULTS.json',dict(status='COMPLETE',timestamp=utc_now(),feature=j,fold=fold,inner=inner,
                          selection_split=split,mode='EARLY_SELECT',scope=scope.provenance(),records=result,
                          original_model_sha256=digest(original/'MODEL.json'),new_estimator_fit_calls=0,
                          validation_auxiliary_time_maximum=int((times[va]+16).max()),
                          late_auxiliary_targets_labels_or_values_used_for_this_selection=False,
                          late_stress_is_not_an_early_candidate_selection_input=True))
                    save_npz(out/'GROUP_LOSSES.npz',group_id=gids,loss=loss)
                    atomic(out/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),files_sha256={n:digest(out/n) for n in ['RESULTS.json','GROUP_LOSSES.npz']}))
                    receipts.append(dict(name=name,done_sha256=digest(out/'DONE.json')))
                atomic(output/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),feature=j,selection_panels=receipts,
                      new_estimator_fit_calls=0,implementation_lock_sha256=digest(BASE/'EARLY_SELECTION_IMPLEMENTATION_LOCK_NEXT9.json')))
        assert not violations
        atomic(BASE/'early_selection'/f'PART_{part}_DONE.json',dict(status='COMPLETE',timestamp=utc_now(),features=features,
              no_break_label_or_tau_read=True,NPZ_members=sorted(set(accesses)),boundary_violations=violations))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--part',type=int,choices=range(4),required=True)
    main(p.parse_args().part)
