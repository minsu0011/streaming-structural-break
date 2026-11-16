"""Original outer A evaluation, separated from current break-label quality."""
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
        from src.next9.domains import masks
        from src.next9.movement_screen import auxiliary_targets,group_weights,grouped_means
        from src.next9.states import directions
        from src.next9.inference import weights_for,interval
        from src.next9.io import save_npz
        bank=FeatureBank();scope=outer_scope(bank,fold);root=BASE/'outer_movement_evaluation'/scope.name
        if (root/'DONE.json').exists():return
        closure=BASE/'outer_movement_refits'/scope.name/'DONE.json'
        assert digest(closure)==global_freeze['source_receipts_sha256'][closure.relative_to(BASE).as_posix()]
        manifest=read_json(closure);mode_records=[]
        for mode in ['NORMAL','EARLY']:
            source=closure.parent/mode
            member=next(r for r in manifest['modes'] if r['mode']==mode)
            assert digest(source/'DONE.json')==member['sha256'];features=read_json(source/'DONE.json')['features']
            global_member=next(r for r in global_freeze['records'] if r['fold']==fold and r['mode']==mode)
            domain_receipt=BASE/global_member['path'];assert digest(domain_receipt)==global_member['sha256']
            domains=read_json(domain_receipt)['domain_spec'];feature_records=[]
            tr=bank.forecast_rows(bank.sampled_rows(scope.training_ids),16,cutoff=199 if mode=='EARLY' else None)
            train_groups=np.asarray([bank.guard.split.groups[int(s)] for s in bank.row_ids[tr]],dtype=np.int32)
            tw=group_weights(train_groups)
            for f in features:
                j=f['feature'];directory=source/f'F{j:03d}';out=root/mode/f'F{j:03d}'
                if (out/'DONE.json').exists():feature_records.append(dict(feature=j,sha256=digest(out/'DONE.json')));continue
                assert digest(directory/'DONE.json')==f['sha256'];done=read_json(directory/'DONE.json')
                assert digest(directory/'FROZEN.json')==done['frozen_sha256'];frozen=read_json(directory/'FROZEN.json')
                assert digest(directory/'PREDICTIONS.npz')==frozen['files_sha256']['PREDICTIONS.npz']
                with np.load(directory/'PREDICTIONS.npz') as z:
                    rows=z['global_row'];pre=z['pre_direction'];pred={k:z[k] for k in ['D1','D2','D3','D5']}
                with activity(f'{scope.name} {mode} F{j:03d} TRAIN fixed D0C smoothing and outer auxiliary targets',
                              'OUTER_AUXILIARY_EVALUATION','B5_SELF_DYNAMICS_VS_FILTER_MEMORY'):
                    ts=bank.pre_state(tr,j);ty=auxiliary_targets(bank,tr,j,ts['S1_DIRECTION'])
                    tc=bank.current(tr,[j])[:,0]
                    td=directions(tc-bank.history(tr,j)[:,-1],bank.series['epsilon'][bank.series_index[tr],0,j])
                    stay=[float(np.clip(np.average(np.argmax(ty[:,5*k+1:5*k+4],axis=1)==td,weights=tw),1/3+.001,.99)) for k in range(3)]
                    reverse_prior=[float(np.clip(np.average(ty[:,5*k+4],weights=tw),1e-6,1-1e-6)) for k in range(3)]
                    smoothing=dict(status='FROZEN',scope=scope.provenance(),stay=stay,reversal_prior=reverse_prior,
                        outer_auxiliary_targets_used=False,break_labels_used=False)
                    if (out/'D0C_TRAIN_FROZEN.json').exists():assert read_json(out/'D0C_TRAIN_FROZEN.json')==smoothing
                    else:atomic(out/'D0C_TRAIN_FROZEN.json',smoothing)
                    y=auxiliary_targets(bank,rows,j,pre);current=bank.current(rows,[j])[:,0]
                    cd=directions(current-bank.history(rows,j)[:,-1],bank.series['epsilon'][bank.series_index[rows],0,j])
                    groups=np.asarray([bank.guard.split.groups[int(s)] for s in bank.row_ids[rows]],dtype=np.int32)
                    names=[];loss=[]
                    for k,h in enumerate([1,4,16]):
                        col=5*k;target=y[:,col];cls=np.argmax(y[:,col+1:col+4],axis=1);rev=y[:,col+4]
                        baseline_probability=np.where(cd==cls,stay[k],(1-stay[k])/2)
                        fixed={'D0A_MSE':(target-current)**2,'D0B_MSE':target**2,'D0C_DIRECTION_LOGLOSS':-np.log(baseline_probability),
                            'D0C_DIRECTION_ERROR':(cd!=cls).astype(float),
                            'TRAIN_REVERSAL_LOGLOSS':-rev*np.log(reverse_prior[k])-(1-rev)*np.log1p(-reverse_prior[k])}
                        for method,p in pred.items():
                            probs=np.clip(p[:,col+1:col+4],1e-6,1.);probs=probs/probs.sum(axis=1,keepdims=True)
                            rp=np.clip(p[:,col+4],1e-6,1-1e-6)
                            fixed.update({method+'_MSE':(target-np.clip(p[:,col],-12,12))**2,
                                method+'_DIRECTION_LOGLOSS':-np.log(probs[np.arange(len(cls)),cls]),
                                method+'_DIRECTION_ERROR':(probs.argmax(axis=1)!=cls).astype(float),
                                method+'_REVERSAL_LOGLOSS':-rev*np.log(rp)-(1-rev)*np.log1p(-rp)})
                        for name,value in fixed.items():names.append(f'h{h}_{name}');loss.append(value)
                    loss=np.column_stack(loss)
                domain_records=[]
                for family,name,keep in masks(bank,rows,domains):
                    dest=out/name
                    if (dest/'DONE.json').exists():domain_records.append(dict(domain=name,sha256=digest(dest/'DONE.json')));continue
                    with activity(f'{scope.name} {mode} F{j:03d} {name} paired whole-group A intervals',
                                  'OUTER_AUXILIARY_EVALUATION','B5_SELF_DYNAMICS_VS_FILTER_MEMORY'):
                        unique=np.unique(groups[keep]);contrasts={};points={}
                        if len(unique)<64:
                            status='INSUFFICIENT_SUPPORT';boot=np.full((2000,len(names)),np.nan);resampling=dict(draws=2000)
                        else:
                            status='SUPPORTED';unique,mean=grouped_means(loss[keep],groups[keep])
                            weights,resampling=weights_for(unique,f'OUTER_A|{scope.name}|{mode}|F{j}|{name}')
                            point=mean.mean(axis=0);boot=(weights@mean)/weights.sum(axis=1)[:,None]
                            points=dict(zip(names,map(float,point)))
                            index={v:i for i,v in enumerate(names)}
                            for h in [1,4,16]:
                                for candidate in ['D1','D2','D3','D5']:
                                    for reference in ['D0A','D0B','D1','D3','D5']:
                                        if candidate==reference:continue
                                        a,b=index[f'h{h}_{candidate}_MSE'],index[f'h{h}_{reference}_MSE']
                                        contrasts[f'h{h}_{candidate}__vs__{reference}_MSE']=interval(point[b]-point[a],boot[:,b]-boot[:,a])
                                    for metric,reference in [('DIRECTION_LOGLOSS','D0C'),('REVERSAL_LOGLOSS','TRAIN')]:
                                        a,b=index[f'h{h}_{candidate}_{metric}'],index[f'h{h}_{reference}_{metric}']
                                        contrasts[f'h{h}_{candidate}__vs__{reference}_{metric}']=interval(point[b]-point[a],boot[:,b]-boot[:,a])
                        save_npz(dest/'BOOTSTRAP.npz',values=boot)
                        atomic(dest/'DONE.json',dict(status='COMPLETE',inference_status=status,timestamp=utc_now(),fold=fold,mode=mode,
                            feature=j,family=family,domain=name,rows=int(keep.sum()),groups=len(unique),points=points,contrasts=contrasts,
                            bootstrap_columns=names,bootstrap_sha256=digest(dest/'BOOTSTRAP.npz'),resampling=resampling,
                            D4_alias='D3',current_F_in_D0A_D0C_D5_not_D1=True,break_labels_used=False))
                        domain_records.append(dict(domain=name,sha256=digest(dest/'DONE.json')))
                atomic(out/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),scope=scope.provenance(),feature=j,mode=mode,domains=domain_records))
                feature_records.append(dict(feature=j,sha256=digest(out/'DONE.json')))
            atomic(root/mode/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),scope=scope.provenance(),mode=mode,features=feature_records))
            mode_records.append(dict(mode=mode,sha256=digest(root/mode/'DONE.json')))
        assert not violations
        atomic(root/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),scope=scope.provenance(),modes=mode_records,
            NPZ_members=sorted(set(accesses)),global_freeze_sha256=digest(global_path),break_labels_or_tau_used=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',required=True,type=int,choices=range(5));main(p.parse_args().fold)
