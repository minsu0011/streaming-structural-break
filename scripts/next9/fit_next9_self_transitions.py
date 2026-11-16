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
    lock=read_json(BASE/'SELF_TRANSITION_IMPLEMENTATION_LOCK_NEXT9.json')
    for name,sha in lock['source_sha256'].items(): assert digest(ROOT/name)==sha,name
    with target_free_npz() as accesses:
        import numpy as np
        import joblib
        from src.next9.data import FeatureBank,outer_scope,inner_scope,selection_splits
        from src.next9.transitions import fit_law
        from src.next9.transition_inputs import probabilities
        from src.next9.movement_screen import paired_group_summary,grouped_means
        from src.next9.io import save_npz
        bank=FeatureBank();parent=outer_scope(bank,fold) if inner is None else inner_scope(bank,fold,inner)
        candidate=read_json(BASE/'candidates'/parent.name/'FROZEN.json')
        root=BASE/'self_transitions'/parent.name
        if (root/'DONE.json').exists():return
        with np.load(BASE/'state_cache/INDEX.npz') as z:rows=z['global_row']
        # Auxiliary next pre-state depends through F[t], yet its registered
        # timestamp is t+1; strict early source and auxiliary cutoff both hold.
        keep=np.isin(rows,bank.forecast_rows(rows,1));rows=rows[keep]
        ids=bank.row_ids[rows];times=bank.times[rows]
        groups=np.asarray([bank.guard.split.groups[int(s)] for s in ids],dtype=np.int32)
        context=bank.series['context'][bank.series_index[rows]]
        variants=['T0_PRIOR','T1_SELF_MARKOV','T2_SELF_RUN_SLOPE_H','T3_SAME_INPUT_DEPTH3']
        allrecords=[]
        for mode,key in [('NORMAL','normal_candidates'),('EARLY','strict_early_candidates')]:
            selected=[]
            for j in candidate[key]:
                out=root/mode/f'F{j:03d}'
                if (out/'DONE.json').exists():
                    selected.append(read_json(out/'DONE.json'));continue
                with activity(f'{parent.name} {mode} F{j:03d} four self transition laws','SELF_TRANSITION','B5_SELF_DYNAMICS_VS_FILTER_MEMORY'):
                    states=bank.pre_state(rows,j);target=bank.pre_state(rows+1,j)['S1_DIRECTION']
                    panels=[]
                    for split,scope in enumerate(selection_splits(bank,fold,inner)):
                        cut=(times+1<=199) if mode=='EARLY' else np.ones(len(rows),dtype=bool)
                        tr=np.flatnonzero(cut & np.isin(ids,list(scope.training_ids)))
                        va=np.flatnonzero(cut & np.isin(ids,list(scope.validation_ids)))
                        losses={};files={};records={}
                        for variant in variants:
                            model=fit_law(states['S1_DIRECTION'][tr],target[tr],states['direction_run_capped15'][tr],states['slope8'][tr],context[tr],ids[tr],groups[tr],scope,variant)
                            pred=model.probabilities(states['S1_DIRECTION'][va],states['direction_run_capped15'][va],states['slope8'][va],context[va])
                            losses[variant]=-np.log(np.maximum(pred[np.arange(len(va)),target[va]],1e-8))
                            p=out/f'panel_{split}'/(variant+'.joblib');p.parent.mkdir(parents=True,exist_ok=True)
                            joblib.dump(model,p);files[p.name]=digest(p)
                            records[variant]=dict(mean_logloss=float(grouped_means(losses[variant],groups[va])[1].mean()),training=model.training,table=model.table.tolist())
                        contrasts={v+'__vs__'+r:paired_group_summary(losses[r],losses[v],groups[va]) for v,r in [
                            ('T1_SELF_MARKOV','T0_PRIOR'),('T2_SELF_RUN_SLOPE_H','T0_PRIOR'),('T2_SELF_RUN_SLOPE_H','T1_SELF_MARKOV'),('T2_SELF_RUN_SLOPE_H','T3_SAME_INPUT_DEPTH3')]}
                        gids,means=grouped_means(np.column_stack([losses[v] for v in variants]),groups[va])
                        save_npz(out/f'panel_{split}'/'GROUP_LOSSES.npz',group_id=gids,loss=means)
                        panel=dict(split=split,scope=scope.provenance(),records=records,contrasts=contrasts,files_sha256=files)
                        atomic(out/f'panel_{split}'/'DONE.json',panel);panels.append(panel)
                    richer=sum(all(p['contrasts']['T2_SELF_RUN_SLOPE_H__vs__'+ref]['ci_low']>0 for ref in ['T0_PRIOR','T1_SELF_MARKOV']) for p in panels)>=2
                    markov=sum(p['contrasts']['T1_SELF_MARKOV__vs__T0_PRIOR']['ci_low']>0 for p in panels)>=2
                    chosen='T2_SELF_RUN_SLOPE_H' if richer else 'T1_SELF_MARKOV' if markov else 'T0_PRIOR'
                    tr=np.flatnonzero(np.isin(ids,list(parent.training_ids)) & ((times+1<=199) if mode=='EARLY' else True))
                    model=fit_law(states['S1_DIRECTION'][tr],target[tr],states['direction_run_capped15'][tr],states['slope8'][tr],context[tr],ids[tr],groups[tr],parent,chosen)
                    joblib.dump(model,out/'LAW.joblib')
                    done=dict(status='COMPLETE',timestamp=utc_now(),feature=j,mode=mode,scope=parent.provenance(),chosen_variant=chosen,
                        self_transition_increment_supported=chosen!='T0_PRIOR',panels=panels,law_sha256=digest(out/'LAW.joblib'),
                        training=model.training,law_table=model.table.tolist(),selection_early_no_late_outcomes=mode=='EARLY',
                        branch_in_M5_requires_separately_frozen_orientation_rule=True,
                        candidate_freeze_sha256=digest(BASE/'candidates'/parent.name/'FROZEN.json'))
                    atomic(out/'DONE.json',done);selected.append(done)
            atomic(root/mode/'DONE.json',dict(status='COMPLETE',scope=parent.provenance(),mode=mode,records=selected))
            allrecords.append(dict(mode=mode,features=candidate[key],sha256=digest(root/mode/'DONE.json')))
        assert not violations
        atomic(root/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),scope=parent.provenance(),records=allrecords,
            no_break_labels_or_tau=True,NPZ_members=sorted(set(accesses)),implementation_sha256=digest(BASE/'SELF_TRANSITION_IMPLEMENTATION_LOCK_NEXT9.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,required=True,choices=range(5));p.add_argument('--inner',type=int,choices=range(3))
    a=p.parse_args();main(a.fold,a.inner)
