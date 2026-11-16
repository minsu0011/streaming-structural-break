"""Held-out exact-parent and mechanical-control gains, without edge selection."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now
from src.next9.boundary import install_workspace_boundary,target_free_npz


def main(fold):
    violations=install_workspace_boundary()
    lock=read_json(BASE/'OUTER_EVALUATION_IMPLEMENTATION_LOCK_NEXT9.json')
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    global_path=BASE/'GLOBAL_PREDICTIONS_FROZEN_NEXT9.json';global_freeze=read_json(global_path)
    assert global_freeze['status']=='FROZEN'
    with target_free_npz() as accesses:
        import numpy as np
        from src.next9.data import FeatureBank,outer_scope
        from src.next9.movement_screen import auxiliary_targets,grouped_means
        from src.next9.inference import weights_for,interval
        from src.next9.io import save_npz
        bank=FeatureBank();scope=outer_scope(bank,fold);root=BASE/'outer_cross_evaluation'/scope.name
        if (root/'DONE.json').exists():return
        closure=BASE/'outer_cross_controls'/scope.name/'DONE.json'
        assert digest(closure)==global_freeze['source_receipts_sha256'][closure.relative_to(BASE).as_posix()]
        manifest=read_json(closure);modes=[]
        for mode in ['NORMAL','EARLY']:
            source=closure.parent/mode;member=next(r for r in manifest['modes'] if r['mode']==mode)
            assert digest(source/'DONE.json')==member['sha256'];edge_manifest=read_json(source/'DONE.json');records=[]
            for item in edge_manifest['edges']:
                edge=item['edge'];j,p=edge['destination'],edge['parent'];name=f'EDGE_{p}_TO_{j}'
                directory=source/name;out=root/mode/name
                if (out/'DONE.json').exists():records.append(read_json(out/'DONE.json'));continue
                assert digest(directory/'DONE.json')==item['sha256'];frozen=read_json(directory/'DONE.json')
                assert digest(directory/'PREDICTIONS.npz')==frozen['prediction_sha256']
                with activity(f'{scope.name} {mode} {name} frozen outer mechanical and exact-remove-parent contrasts',
                              'OUTER_CROSS_EVALUATION','B5_SELF_DYNAMICS_VS_FILTER_MEMORY'):
                    with np.load(directory/'PREDICTIONS.npz') as z:
                        rows=z['global_row'];pre=z['pre_direction'];pred={k:z[k] for k in ['D3','D5','PARENT','FULL_LAW','REMOVE_PARENT_LAW']}
                    y=auxiliary_targets(bank,rows,j,pre);next_state=bank.pre_state(rows+1,j)['S1_DIRECTION']
                    group=np.asarray([bank.guard.split.groups[int(s)] for s in bank.row_ids[rows]],dtype=np.int32)
                    losses={}
                    for k,h in enumerate([1,4,16]):
                        for method in ['D3','D5','PARENT']:
                            losses[f'h{h}_{method}']=(y[:,5*k]-np.clip(pred[method][:,5*k],-12,12))**2
                    for method in ['FULL_LAW','REMOVE_PARENT_LAW']:
                        losses[method]=-np.log(np.maximum(pred[method][np.arange(len(rows)),next_state],1e-8))
                    names=list(losses);groups,mean=grouped_means(np.column_stack(list(losses.values())),group)
                    weights,resampling=weights_for(groups,f'OUTER_CROSS|{scope.name}|{mode}|{name}')
                    point=mean.mean(axis=0);boot=(weights@mean)/weights.sum(axis=1)[:,None];index={k:i for i,k in enumerate(names)}
                    contrasts={}
                    for h in [1,4,16]:
                        a=index[f'h{h}_PARENT']
                        for method in ['D3','D5']:
                            b=index[f'h{h}_{method}'];contrasts[f'h{h}_PARENT__vs__{method}']=interval(point[b]-point[a],boot[:,b]-boot[:,a])
                    a,b=index['FULL_LAW'],index['REMOVE_PARENT_LAW']
                    contrasts['EXACT_REMOVE_PARENT_LOGLOSS']=interval(point[b]-point[a],boot[:,b]-boot[:,a])
                    chosen=edge['horizon'];keys=[f'h{chosen}_PARENT__vs__D3',f'h{chosen}_PARENT__vs__D5','EXACT_REMOVE_PARENT_LOGLOSS']
                    good=len(groups)>=64 and all(contrasts[k]['status']=='SUPPORTED' and contrasts[k]['ci_low']>0 for k in keys)
                    save_npz(out/'BOOTSTRAP.npz',values=boot)
                    record=dict(status='COMPLETE',timestamp=utc_now(),fold=fold,mode=mode,edge=edge,scope=scope.provenance(),
                        groups=len(groups),rows=len(rows),points=dict(zip(names,map(float,point))),contrasts=contrasts,
                        TRAIN_selected_horizon=chosen,no_outer_horizon_selection=True,passes_frozen_horizon=bool(good),
                        bootstrap_columns=names,bootstrap_sha256=digest(out/'BOOTSTRAP.npz'),resampling=resampling,
                        causal_parent_claim=False,break_label_or_tau_used=False)
                    atomic(out/'DONE.json',record);records.append(record)
            atomic(root/mode/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),scope=scope.provenance(),mode=mode,records=records,
                zero_edge_branch=edge_manifest['zero_edge_branch'],no_new_edges=True))
            modes.append(dict(mode=mode,sha256=digest(root/mode/'DONE.json')))
        assert not violations
        atomic(root/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),scope=scope.provenance(),modes=modes,
            global_freeze_sha256=digest(global_path),NPZ_members=sorted(set(accesses)),break_labels_or_tau_used=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',required=True,type=int,choices=range(5));main(p.parse_args().fold)
