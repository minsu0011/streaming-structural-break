"""Fit M1..M5 inside each excluded-group procedure and issue unscored rows."""
from pathlib import Path
import sys
import argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now
from src.next9.boundary import install_workspace_boundary


def main(fold,inner):
    violations=install_workspace_boundary()
    lock=read_json(BASE/'JOINT_HEAD_IMPLEMENTATION_LOCK_V2_NEXT9.json')
    for name,sha in lock['source_sha256'].items(): assert digest(ROOT/name)==sha,name
    import numpy as np
    import joblib
    from src.next9.data import FeatureBank,outer_scope,inner_scope
    from src.next9.labels import ScopedLabels
    from src.next9.orientation_contract import fit_common_range,fit_run_fallback
    from src.next9.joint_design import assemble,alias_map
    from src.next9.learning_converged_v2 import fit_classifier,continue_classifier
    from src.next9.io import save_npz
    bank=FeatureBank();parent=outer_scope(bank,fold) if inner is None else inner_scope(bank,fold,inner)
    labels=ScopedLabels(bank)
    root=BASE/'joint_heads'/parent.name
    if (root/'DONE.json').exists():return
    records=[]
    for mode in ['NORMAL','EARLY']:
        out=root/mode
        if (out/'DONE.json').exists():
            records.append(dict(mode=mode,sha256=digest(out/'DONE.json')));continue
        if out.exists() and any(out.iterdir()): raise RuntimeError('Partial joint head preserved for explicit recovery')
        selection=read_json(BASE/'orientation_selection'/parent.name/mode/'SELECTION_FROZEN.json')
        assert selection['scope']==parent.provenance()
        transition=read_json(BASE/'self_transitions'/parent.name/mode/'DONE.json')
        with activity(f'{parent.name} {mode} TRAIN-fixed M1 toM5 and unscored full-prefix predictions','JOINT_MODELS','B6_BL80_REDUNDANCY'):
            tr=bank.sampled_rows(parent.training_ids,cutoff=199 if mode=='EARLY' else None)
            va=bank.sampled_rows(parent.validation_ids,stride=1)
            y=labels.get(tr,parent,'TRAIN',early=mode=='EARLY')
            ids=bank.row_ids[tr];groups=np.asarray([bank.guard.split.groups[int(s)] for s in ids],dtype=np.int32)
            candidates=selection['candidates'];run_maps={};rule_specs={};laws={};abstained=[]
            for j in candidates:
                state=bank.pre_state(tr,j)
                run_maps[j]=fit_run_fallback(state['S2_DIRECTION_RUN'],ids,parent)
                rule=next((r for r in selection['provisional_rules'] if r['feature']==j),None)
                if rule is None: continue
                domain=fit_common_range(bank.current(tr,[j])[:,0],state[rule['pair'][0]],ids,y,parent,tuple(rule['pair']))
                if domain.status!='SUPPORTED':
                    abstained.append(dict(feature=j,reason='Full TRAIN common-range support failed',domain=domain.record()));continue
                rule_specs[j]={**rule,'low':domain.low,'high':domain.high,'domain':domain.record()}
                record=next(r for r in transition['records'] if r['feature']==j)
                if record['self_transition_increment_supported']:
                    p=BASE/'self_transitions'/parent.name/mode/f'F{j:03d}'/'LAW.joblib'
                    assert digest(p)==record['law_sha256'];laws[j]=joblib.load(p)
            aliases=alias_map(candidates,rule_specs,laws)
            full,full_names=assemble(bank,tr,candidates,rule_specs,run_maps,laws,'M5')
            methods=list(dict.fromkeys(aliases.values()));columns={};models={};files={}
            out.mkdir(parents=True,exist_ok=True)
            for method in methods:
                _,names=assemble(bank,tr[:2],candidates,rule_specs,run_maps,laws,method)
                indices=np.asarray([full_names.index(n) for n in names],dtype=int);columns[method]=indices
                previous=BASE/'repairs/solver_cap250_preserved/joint_heads'/parent.name/mode
                checkpoint=previous/(method+'.joblib')
                if checkpoint.exists():
                    oldspec=read_json(previous/'SPEC_FROZEN.json')
                    assert digest(checkpoint)==oldspec['files_sha256'][checkpoint.name]
                    model=joblib.load(checkpoint)
                    assert model.training['input_names']==list(names) and model.training['feature_candidates']==candidates
                    assert oldspec['rule_specs']=={str(k):v for k,v in rule_specs.items()}
                    model=continue_classifier(model,full[:,indices],y,ids,groups,parent)
                    model.training['preserved_cap_checkpoint_sha256']=digest(checkpoint)
                else:
                    model=fit_classifier(full[:,indices],y,ids,groups,parent)
                model.training.update(model_method=method,early_training_maximum=199 if mode=='EARLY' else None,
                    input_names=list(names),feature_candidates=candidates,orientation_rule_features=list(rule_specs),self_transition_features=list(laws),
                    selection_sha256=digest(BASE/'orientation_selection'/parent.name/mode/'SELECTION_FROZEN.json'),
                    validation_labels_used=False)
                models[method]=model
                joblib.dump(model,out/(method+'.joblib'));files[method+'.joblib']=digest(out/(method+'.joblib'))
            del full
            atomic(out/'SPEC_FROZEN.json',dict(status='FROZEN',timestamp=utc_now(),scope=parent.provenance(),mode=mode,
                candidates=candidates,rule_specs=rule_specs,run_maps={j:m.tolist() for j,m in run_maps.items()},
                law_features=list(laws),abstained=abstained,aliases=aliases,input_names=list(full_names),
                selected_columns={m:c.tolist() for m,c in columns.items()},files_sha256=files,validation_labels_used=False))
            predictions={m:np.empty(len(va),dtype=np.float32) for m in methods}
            for start in range(0,len(va),32768):
                sl=slice(start,start+32768)
                x,names=assemble(bank,va[sl],candidates,rule_specs,run_maps,laws,'M5')
                assert names==full_names
                for method in methods:
                    predictions[method][sl]=models[method].predict_proba(x[:,columns[method]])[:,1]
            save_npz(out/'PREDICTIONS.npz',global_row=va,dataset_id=bank.row_ids[va],**{m:predictions[a] for m,a in aliases.items()})
            files['PREDICTIONS.npz']=digest(out/'PREDICTIONS.npz');files['SPEC_FROZEN.json']=digest(out/'SPEC_FROZEN.json')
            atomic(out/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),scope=parent.provenance(),mode=mode,
                files_sha256=files,aliases=aliases,training_rows=len(tr),prediction_rows=len(va),
                estimator_fit_calls=sum(m.training['estimator_fit_calls'] for m in models.values()),
                eligible_candidates=candidates,provisional_rule_count=len(rule_specs),self_law_count=len(laws),
                same_time_full_row_predictions=True,outer_scores_released=False))
            records.append(dict(mode=mode,sha256=digest(out/'DONE.json')))
    assert not violations
    atomic(root/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),scope=parent.provenance(),records=records,
        label_access=labels.accesses,outer_scores_released=False,implementation_sha256=digest(BASE/'JOINT_HEAD_IMPLEMENTATION_LOCK_V2_NEXT9.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,required=True,choices=range(5));p.add_argument('--inner',type=int,choices=range(3))
    a=p.parse_args();main(a.fold,a.inner)
