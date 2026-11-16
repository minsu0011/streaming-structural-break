"""Outer auxiliary comparison of the four already registered self laws."""
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
    lock=read_json(BASE/'OUTER_SELF_DIAGNOSTIC_LOCK_NEXT9.json')
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    assert read_json(BASE/'GLOBAL_PREDICTIONS_FROZEN_NEXT9.json')['status']=='FROZEN'
    with target_free_npz() as accesses:
        import numpy as np
        import joblib
        from src.next9.data import FeatureBank,outer_scope
        from src.next9.transitions import fit_law
        from src.next9.transition_inputs import probabilities
        from src.next9.movement_screen import grouped_means,group_weights
        from src.next9.inference import weights_for,interval
        from src.next9.domains import masks
        from src.next9.io import save_npz
        bank=FeatureBank();scope=outer_scope(bank,fold);root=BASE/'outer_self_transition_evaluation'/scope.name
        if (root/'DONE.json').exists():return
        variants=['T0_PRIOR','T1_SELF_MARKOV','T2_SELF_RUN_SLOPE_H','T3_SAME_INPUT_DEPTH3'];modes=[]
        for mode in ['NORMAL','EARLY']:
            selection=read_json(BASE/'self_transitions'/scope.name/mode/'DONE.json')
            assert selection['scope']==scope.provenance()
            tr=bank.forecast_rows(bank.sampled_rows(scope.training_ids),1,cutoff=199 if mode=='EARLY' else None)
            va=bank.forecast_rows(bank.sampled_rows(scope.validation_ids,start=400 if mode=='EARLY' else 0),1)
            ids=bank.row_ids[tr];tg=np.asarray([bank.guard.split.groups[int(s)] for s in ids],dtype=np.int32)
            vg=np.asarray([bank.guard.split.groups[int(s)] for s in bank.row_ids[va]],dtype=np.int32)
            tc=bank.series['context'][bank.series_index[tr]];vc=bank.series['context'][bank.series_index[va]]
            domains=read_json(BASE/'frozen_predictions'/scope.name/mode/'FROZEN.json')['domain_spec'];records=[]
            for chosen in selection['records']:
                j=chosen['feature'];out=root/mode/f'F{j:03d}'
                if (out/'DONE.json').exists():records.append(dict(feature=j,sha256=digest(out/'DONE.json')));continue
                if out.exists() and any(out.iterdir()):raise RuntimeError('Partial self diagnostic evidence preserved for explicit recovery')
                with activity(f'{scope.name} {mode} F{j:03d} fixed four self-law diagnostic refits before auxiliary outer evaluation',
                              'OUTER_SELF_TRANSITION_DIAGNOSTIC','B5_SELF_DYNAMICS_VS_FILTER_MEMORY'):
                    ts=bank.pre_state(tr,j);vs=bank.pre_state(va,j);target=bank.pre_state(tr+1,j)['S1_DIRECTION']
                    predictions={};files={};models={};out.mkdir(parents=True,exist_ok=True)
                    for variant in variants:
                        if variant==chosen['chosen_variant']:
                            path=BASE/'self_transitions'/scope.name/mode/f'F{j:03d}'/'LAW.joblib'
                            assert digest(path)==chosen['law_sha256'];model=joblib.load(path)
                        else:model=fit_law(ts['S1_DIRECTION'],target,ts['direction_run_capped15'],ts['slope8'],tc,ids,tg,scope,variant)
                        models[variant]=model;predictions[variant]=probabilities(model,vs,vc)
                        joblib.dump(model,out/(variant+'.joblib'));files[variant+'.joblib']=digest(out/(variant+'.joblib'))
                    save_npz(out/'PREDICTIONS.npz',global_row=va,**predictions);files['PREDICTIONS.npz']=digest(out/'PREDICTIONS.npz')
                    atomic(out/'PREDICTIONS_FROZEN.json',dict(status='FROZEN',timestamp=utc_now(),scope=scope.provenance(),feature=j,mode=mode,
                        files_sha256=files,chosen_variant=chosen['chosen_variant'],outer_auxiliary_targets_used=False,
                        no_break_labels_used=True,diagnostic_only_no_changes_to_M_models=True))
                    y=bank.pre_state(va+1,j)['S1_DIRECTION'];ix=np.arange(len(va))
                    loss=np.column_stack([-np.log(np.maximum(predictions[v][ix,y],1e-8)) for v in variants])
                    matrices={};w=group_weights(vg)
                    for name,prediction in predictions.items():
                        rows=[]
                        for state in range(4):
                            keep=vs['S1_DIRECTION']==state
                            rows.append(np.average(prediction[keep],axis=0,weights=w[keep]).tolist() if keep.any() else [None]*4)
                        matrices[name]=rows
                    observed=[]
                    for state in range(4):
                        keep=vs['S1_DIRECTION']==state
                        observed.append((np.bincount(y[keep],weights=w[keep],minlength=4)/w[keep].sum()).tolist() if keep.any() else [None]*4)
                    atomic(out/'TRANSITION_MATRICES.json',dict(predicted_on_heldout_covariates=matrices,observed_heldout=observed,
                        weighting='Equal original group total weight, then condition on pre-current state',chosen_variant=chosen['chosen_variant'],
                        TRAIN_markov_table=models['T1_SELF_MARKOV'].table.tolist(),not_claiming_state_is_complete_Markov_memory=True))
                domain_records=[]
                for family,name,keep in masks(bank,va,domains):
                    dest=out/name
                    with activity(f'{scope.name} {mode} F{j:03d} {name} paired self transition logloss intervals',
                                  'OUTER_SELF_TRANSITION_DIAGNOSTIC','B5_SELF_DYNAMICS_VS_FILTER_MEMORY'):
                        groups=np.unique(vg[keep]);points={};contrasts={};boot=np.full((2000,4),np.nan);resampling=dict(draws=2000)
                        if len(groups)>=64:
                            groups,mean=grouped_means(loss[keep],vg[keep]);weights,resampling=weights_for(groups,f'OUTER_T|{scope.name}|{mode}|F{j}|{name}')
                            point=mean.mean(axis=0);boot=(weights@mean)/weights.sum(axis=1)[:,None];points=dict(zip(variants,map(float,point)))
                            for a,b in [(1,0),(2,0),(2,1),(2,3)]:
                                contrasts[variants[a]+'__vs__'+variants[b]]=interval(point[b]-point[a],boot[:,b]-boot[:,a])
                        save_npz(dest/'BOOTSTRAP.npz',values=boot)
                        atomic(dest/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),fold=fold,mode=mode,feature=j,family=family,domain=name,
                            rows=int(keep.sum()),groups=len(groups),inference_status='SUPPORTED' if len(groups)>=64 else 'INSUFFICIENT_SUPPORT',
                            mean_logloss=points,contrasts=contrasts,variants=variants,bootstrap_sha256=digest(dest/'BOOTSTRAP.npz'),resampling=resampling))
                        domain_records.append(dict(domain=name,sha256=digest(dest/'DONE.json')))
                atomic(out/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),scope=scope.provenance(),feature=j,mode=mode,
                    domains=domain_records,chosen_variant=chosen['chosen_variant'],chosen_rule_unchanged=True,
                    prediction_freeze_sha256=digest(out/'PREDICTIONS_FROZEN.json'),matrices_sha256=digest(out/'TRANSITION_MATRICES.json')))
                records.append(dict(feature=j,sha256=digest(out/'DONE.json')))
            atomic(root/mode/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),scope=scope.provenance(),mode=mode,features=records))
            modes.append(dict(mode=mode,sha256=digest(root/mode/'DONE.json')))
        assert not violations
        atomic(root/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),scope=scope.provenance(),modes=modes,
            NPZ_members=sorted(set(accesses)),no_break_labels_used=True,no_scientific_model_changes=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',required=True,type=int,choices=range(5));main(p.parse_args().fold)
