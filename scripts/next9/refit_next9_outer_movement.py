"""Final TRAIN-only A forecast fits and unscored original outer predictions."""
from pathlib import Path
import sys
import argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now
from src.next9.boundary import install_workspace_boundary,target_free_npz


def main(fold):
    violations=install_workspace_boundary()
    lock=read_json(BASE/'OUTER_MOVEMENT_REFIT_LOCK_NEXT9.json')
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    with target_free_npz() as accesses:
        import numpy as np
        from src.next9.data import FeatureBank,outer_scope
        from src.next9.movement_screen import feature_inputs,auxiliary_targets,fit_ridge_bundle
        from src.next9.detailed_movement import design,fit
        from src.next9.io import save_npz
        bank=FeatureBank();scope=outer_scope(bank,fold)
        candidate=read_json(BASE/'candidates'/scope.name/'FROZEN.json')
        root=BASE/'outer_movement_refits'/scope.name
        if (root/'DONE.json').exists():return
        modes=[]
        for mode,key in [('NORMAL','normal_candidates'),('EARLY','strict_early_candidates')]:
            tr=bank.forecast_rows(bank.sampled_rows(scope.training_ids),16,cutoff=199 if mode=='EARLY' else None)
            va=bank.forecast_rows(bank.sampled_rows(scope.validation_ids,start=400 if mode=='EARLY' else 0),16)
            ids=bank.row_ids[tr];groups=np.asarray([bank.guard.split.groups[int(s)] for s in ids],dtype=np.int32)
            records=[]
            for j in candidate[key]:
                out=root/mode/f'F{j:03d}'
                if (out/'DONE.json').exists():records.append(dict(feature=j,sha256=digest(out/'DONE.json')));continue
                if out.exists() and any(out.iterdir()):raise RuntimeError('Partial outer movement refit preserved')
                with activity(f'{scope.name} {mode} F{j:03d} final D1 D2 D3 D5 fits and unscored heldout forecasts','OUTER_MOVEMENT_REFIT','B5_SELF_DYNAMICS_VS_FILTER_MEMORY'):
                    ts=bank.pre_state(tr,j);vs=bank.pre_state(va,j)
                    target=auxiliary_targets(bank,tr,j,ts['S1_DIRECTION'])
                    tx=feature_inputs(bank,tr,j,ts);vx=feature_inputs(bank,va,j,vs)
                    d1=fit_ridge_bundle(tx,target,ids,groups,scope)
                    predictions={'D1':d1.predict(vx).astype(np.float32)};files={}
                    atomic(out/'D1_MODEL.json',d1.record());files['D1_MODEL.json']=digest(out/'D1_MODEL.json')
                    for method in ['D2','D3','D5']:
                        tx,names=design(bank,tr,j,ts,method);vx,vnames=design(bank,va,j,vs,method);assert names==vnames
                        model=fit(tx,target,ids,groups,scope,names,method)
                        predictions[method]=model.predict(vx).astype(np.float32)
                        atomic(out/(method+'_MODEL.json'),model.record());files[method+'_MODEL.json']=digest(out/(method+'_MODEL.json'))
                    save_npz(out/'PREDICTIONS.npz',global_row=va,dataset_id=bank.row_ids[va],pre_direction=vs['S1_DIRECTION'],**predictions)
                    files['PREDICTIONS.npz']=digest(out/'PREDICTIONS.npz')
                    atomic(out/'FROZEN.json',dict(status='FROZEN',timestamp=utc_now(),scope=scope.provenance(),feature=j,mode=mode,
                        files_sha256=files,D4_alias='D3',horizons=[1,4,16],training_rows=len(tr),prediction_rows=len(va),
                        training_auxiliary_max_time=int((bank.times[tr]+16).max()),validation_auxiliary_targets_used=False,
                        break_labels_or_tau_used=False,D1_current_F_excluded=True,D5_current_F_included_as_stronger_control=True,estimator_fit_calls=4))
                    atomic(out/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),scope=scope.provenance(),feature=j,
                        frozen_sha256=digest(out/'FROZEN.json'),files_sha256=files))
                    records.append(dict(feature=j,sha256=digest(out/'DONE.json')))
            atomic(root/mode/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),scope=scope.provenance(),mode=mode,features=records))
            modes.append(dict(mode=mode,sha256=digest(root/mode/'DONE.json')))
        assert not violations
        atomic(root/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),scope=scope.provenance(),modes=modes,
            break_label_tau_read=False,NPZ_members=sorted(set(accesses)),implementation_sha256=digest(BASE/'OUTER_MOVEMENT_REFIT_LOCK_NEXT9.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',required=True,type=int,choices=range(5));main(p.parse_args().fold)
