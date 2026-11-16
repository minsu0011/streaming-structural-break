from pathlib import Path
import sys
import argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now
from src.next9.boundary import install_workspace_boundary,target_free_npz


def main(fold,inner):
    violations=install_workspace_boundary()
    lock=read_json(BASE/'DETAILED_MOVEMENT_IMPLEMENTATION_LOCK_NEXT9.json')
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    with target_free_npz() as accesses:
        import numpy as np
        from src.next9.data import FeatureBank,outer_scope,inner_scope,selection_splits
        from src.next9.detailed_movement import design,fit
        from src.next9.movement_screen import auxiliary_targets,evaluate
        from src.next9.states import directions
        from src.next9.io import save_npz
        bank=FeatureBank();parent=outer_scope(bank,fold) if inner is None else inner_scope(bank,fold,inner)
        candidate=read_json(BASE/'candidates'/parent.name/'FROZEN.json')
        output=BASE/'detailed_movement'/parent.name
        if (output/'DONE.json').exists():return
        with np.load(BASE/'state_cache/INDEX.npz') as z:rows=z['global_row']
        ids=bank.row_ids[rows];times=bank.times[rows]
        groups=np.asarray([bank.guard.split.groups[int(s)] for s in ids],dtype=np.int32)
        common=np.isin(rows,bank.forecast_rows(rows,16))
        features=sorted(set(candidate['normal_candidates'])|set(candidate['strict_early_candidates']))
        records=[]
        for j in features:
            with np.load(BASE/'state_cache'/f'F{j:03d}'/'STATES.npz') as z:states={k:z[k] for k in z.files}
            y=np.zeros((len(rows),15),dtype=np.float32)
            y[common]=auxiliary_targets(bank,rows[common],j,states['S1_DIRECTION'][common])
            current=bank.current(rows,[j])[:,0]
            current_direction=directions(current-bank.history(rows,j)[:,-1],bank.series['epsilon'][bank.series_index[rows],0,j])
            for method in ['D2','D3','D5']:
                x,names=design(bank,rows,j,states,method)
                for mode,key in [('NORMAL','normal_candidates'),('EARLY','strict_early_candidates')]:
                    if j not in candidate[key]:continue
                    for split,scope in enumerate(selection_splits(bank,fold,inner)):
                        out=output/mode/f'F{j:03d}'/method/f'panel_{split}'
                        if (out/'DONE.json').exists():
                            records.append(dict(path=(out/'DONE.json').relative_to(output).as_posix(),sha256=digest(out/'DONE.json')));continue
                        with activity(f'{parent.name} {mode} F{j:03d} {method} movement panel{split}','DETAILED_MOVEMENT','B5_SELF_DYNAMICS_VS_FILTER_MEMORY'):
                            tr=np.flatnonzero(common & np.isin(ids,list(scope.training_ids)) & ((times+16<=199) if mode=='EARLY' else True))
                            va=np.flatnonzero(common & np.isin(ids,list(scope.validation_ids)) & ((times>=400) if mode=='EARLY' else True))
                            model=fit(x[tr],y[tr],ids[tr],groups[tr],scope,names,method)
                            result,gids,loss=evaluate(model,x[va],y[va],current[va],current_direction[va],groups[va],y[tr],current_direction[tr],groups[tr])
                            for r in result:r[method+'_MSE']=r.pop('D1_MSE')
                            atomic(out/'MODEL.json',model.record())
                            atomic(out/'RESULTS.json',dict(status='COMPLETE',feature=j,method=method,mode=mode,records=result,
                                training_auxiliary_time_maximum=int((times[tr]+16).max()),scope=scope.provenance(),D4_alias='D3'))
                            save_npz(out/'GROUP_LOSSES.npz',group_id=gids,loss=loss)
                            atomic(out/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),files_sha256={n:digest(out/n) for n in ['MODEL.json','RESULTS.json','GROUP_LOSSES.npz']}))
                            records.append(dict(path=(out/'DONE.json').relative_to(output).as_posix(),sha256=digest(out/'DONE.json')))
                del x
        assert not violations
        atomic(output/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),scope=parent.provenance(),features=features,records=records,
            D4_alias='D3_NO_CERTIFIED_STABLE_CONTEXT',NPZ_members=sorted(set(accesses)),break_label_tau_read=0,
            candidate_freeze_sha256=digest(BASE/'candidates'/parent.name/'FROZEN.json'),implementation_sha256=digest(BASE/'DETAILED_MOVEMENT_IMPLEMENTATION_LOCK_NEXT9.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,required=True,choices=range(5));p.add_argument('--inner',type=int,choices=range(3))
    a=p.parse_args();main(a.fold,a.inner)
