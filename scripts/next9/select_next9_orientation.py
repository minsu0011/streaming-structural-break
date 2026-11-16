"""Nested TRAIN-only orientation selection from previously frozen A candidates."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE, atomic, read_json, digest, activity, utc_now
from src.next9.boundary import install_workspace_boundary


def main(fold, inner):
    violations=install_workspace_boundary()
    lock=read_json(BASE/'ORIENTATION_SELECTION_IMPLEMENTATION_LOCK_NEXT9.json')
    for name,sha in lock['source_sha256'].items():
        assert digest(ROOT/name)==sha,name
    import numpy as np
    import joblib
    from src.next9.data import FeatureBank,outer_scope,inner_scope,selection_splits
    from src.next9.labels import ScopedLabels
    from src.next9.orientation_contract import PAIR_MENU,fit_common_range,fit_run_fallback,orientation_design,one_hot,support,sufficient
    from src.next9.learning import fit_classifier
    from src.next9.inference import compare
    from src.next9.eligibility import holm
    from src.next9.io import save_npz
    bank=FeatureBank()
    parent=outer_scope(bank,fold) if inner is None else inner_scope(bank,fold,inner)
    candidate=read_json(BASE/'candidates'/parent.name/'FROZEN.json')
    assert candidate['scope']==parent.provenance()
    labels=ScopedLabels(bank)
    with np.load(BASE/'state_cache/INDEX.npz') as z: rows=z['global_row']
    ids=bank.row_ids[rows];times=bank.times[rows]
    groups=np.asarray([bank.guard.split.groups[int(s)] for s in ids],dtype=np.int32)
    root=BASE/'orientation_selection'/parent.name
    if (root/'DONE.json').exists():
        assert read_json(root/'DONE.json')['implementation_sha256']==digest(BASE/'ORIENTATION_SELECTION_IMPLEMENTATION_LOCK_NEXT9.json')
        return
    modes=[]
    for mode,key in [('NORMAL','normal_candidates'),('EARLY','strict_early_candidates')]:
        candidates=candidate[key]
        evidence=[]
        for j in candidates:
            with np.load(BASE/'state_cache'/f'F{j:03d}'/'STATES.npz') as z: states={n:z[n] for n in z.files}
            current=bank.current(rows,[j])[:,0]
            prior=bank.history(rows,j)[:,-1]
            s=bank.series_index[rows]
            level=np.where(~bank.series['valid'][s,j],3,np.where(prior<bank.series['low'][s,j],0,np.where(prior>bank.series['high'][s,j],2,1))).astype(np.int8)
            for split,scope in enumerate(selection_splits(bank,fold,inner)):
                out=root/mode/f'F{j:03d}'/f'panel_{split}'
                if (out/'DONE.json').exists():
                    done=read_json(out/'DONE.json')
                    for name,sha in done['files_sha256'].items(): assert digest(out/name)==sha
                    evidence.extend(read_json(out/'PAIR_RESULTS.json')['records'])
                    continue
                if out.exists() and any(out.iterdir()):
                    raise RuntimeError('Partial orientation panel preserved for explicit recovery')
                with activity(f'{parent.name} {mode} F{j:03d} orientation internal panel{split}', 'ORIENTATION_SELECTION','B1_COMMON_CURRENT_RANGE'):
                    cut=(times<=199) if mode=='EARLY' else np.ones(len(rows),dtype=bool)
                    tr=np.flatnonzero(cut & np.isin(ids,list(scope.training_ids)))
                    va=np.flatnonzero(cut & np.isin(ids,list(scope.validation_ids)))
                    y=labels.get(rows[tr],scope,'TRAIN',early=mode=='EARLY')
                    mapping=fit_run_fallback(states['S2_DIRECTION_RUN'][tr],groups[tr],scope)
                    # Group IDs equal the admitted dependency groups; the source
                    # also verifies no dependency group straddles a partition.
                    domains=[fit_common_range(current[tr],states[p[0]][tr],ids[tr],y,scope,p) for p in PAIR_MENU]
                    train_args=(current[tr],states['S1_DIRECTION'][tr],mapping[states['S2_DIRECTION_RUN'][tr]],states['S4_REVERSAL'][tr],level[tr])
                    valid_args=(current[va],states['S1_DIRECTION'][va],mapping[states['S2_DIRECTION_RUN'][va]],states['S4_REVERSAL'][va],level[va])
                    predictions={};models={};files={}
                    out.mkdir(parents=True,exist_ok=True)
                    for method in ['O0','O1','O2_LEVEL','O3_MOVE','O4_MOVE_RUN','O5_MOVE_REV','O6_TABULAR','STATE_ONLY']:
                        tx=orientation_design(*train_args,method);vx=orientation_design(*valid_args,method)
                        model=fit_classifier(tx,y,ids[tr],groups[tr],scope,method='DEPTH3' if method=='O6_TABULAR' else 'LOGISTIC')
                        models[method]=model
                        predictions[method]=model.predict_proba(vx)[:,1].astype(np.float32)
                        joblib.dump(model,out/(method+'.joblib'));files[method+'.joblib']=digest(out/(method+'.joblib'))
                    for method,column,count in [('STATE_ONLY_MOVE','S1_DIRECTION',4),('STATE_ONLY_RUN','S2_DIRECTION_RUN',13)]:
                        train_state=states[column][tr];valid_state=states[column][va]
                        if method=='STATE_ONLY_RUN': train_state=mapping[train_state];valid_state=mapping[valid_state]
                        model=fit_classifier(one_hot(train_state,count),y,ids[tr],groups[tr],scope)
                        models[method]=model;predictions[method]=model.predict_proba(one_hot(valid_state,count))[:,1].astype(np.float32)
                        joblib.dump(model,out/(method+'.joblib'));files[method+'.joblib']=digest(out/(method+'.joblib'))
                    predictions['F_ONLY']=predictions['O1']
                    save_npz(out/'PREDICTIONS.npz',global_row=rows[va],**predictions)
                    files['PREDICTIONS.npz']=digest(out/'PREDICTIONS.npz')
                    atomic(out/'PREDICTIONS_FROZEN.json',dict(status='FROZEN',timestamp=utc_now(),scope=scope.provenance(),files_sha256=files,
                        training_domains=[d.record() for d in domains],run_mapping=mapping.tolist(),validation_labels_used=False))
                    vy=labels.get(rows[va],scope,'INTERNAL_VALIDATION',early=mode=='EARLY')
                    records=[]
                    for pair_index,(pair,domain) in enumerate(zip(PAIR_MENU,domains)):
                        method='O4_MOVE_RUN' if pair[0]=='S2_DIRECTION_RUN' else 'O5_MOVE_REV' if pair[0]=='S4_REVERSAL' else 'O3_MOVE'
                        keep=domain.mask(current[va],states[pair[0]][va])
                        val_support={str(k):support(groups[va],vy,keep & (states[pair[0]][va]==k)) for k in pair[1:]}
                        record=dict(feature=j,split=split,mode=mode,pair_index=pair_index,pair=list(pair),method=method,domain=domain.record(),validation_support=val_support,
                            TRAIN_scope=scope.provenance(),prediction_freeze_sha256=digest(out/'PREDICTIONS_FROZEN.json'))
                        if domain.status!='SUPPORTED' or not all(sufficient(v) for v in val_support.values()):
                            record.update(status='INSUFFICIENT_COMMON_RANGE',difference_p=1.)
                        else:
                            pred={k:v[keep] for k,v in predictions.items() if not k.startswith('STATE_ONLY_')}
                            state_control='STATE_ONLY_RUN' if method=='O4_MOVE_RUN' else 'STATE_ONLY' if method=='O5_MOVE_REV' else 'STATE_ONLY_MOVE'
                            pred['STATE_ONLY']=predictions[state_control][keep]
                            pred['CURRENT_F_DIRECTION']=current[va][keep]
                            result=compare(vy[keep],pred,times[va][keep],groups[va][keep],f'{parent.name}|{mode}|{j}|{split}|pair{pair_index}',direction_states=states[pair[0]][va][keep],pair=pair[1:])
                            record.update(status='COMPLETE',evidence=result,difference_p=result['orientation']['difference']['p_two_sided'],matched_state_only=state_control)
                        records.append(record)
                    atomic(out/'PAIR_RESULTS.json',dict(status='COMPLETE',records=records));files['PAIR_RESULTS.json']=digest(out/'PAIR_RESULTS.json')
                    atomic(out/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),files_sha256=files,
                        estimator_fit_calls=sum(m.training['estimator_fit_calls'] for m in models.values()),
                        implementation_sha256=digest(BASE/'ORIENTATION_SELECTION_IMPLEMENTATION_LOCK_NEXT9.json')))
                    evidence.extend(records)
        # One complete feature-pair family per internal heldout panel, retaining
        # unsupported tests at p1. No family size reduction after outcomes.
        for split in range(3):
            panel=[r for r in evidence if r['split']==split]
            adjusted=holm({(r['feature'],r['pair_index']):r['difference_p'] for r in panel})
            for r in panel:r['difference_Holm_p']=adjusted[(r['feature'],r['pair_index'])]
        selected=[]
        for j in candidates:
            for pair_index,pair in enumerate(PAIR_MENU):
                matched=[r for r in evidence if r['feature']==j and r['pair_index']==pair_index]
                patterns=[]
                for r in matched:
                    if r['status']!='COMPLETE' or r['difference_Holm_p']>=.05: continue
                    e=r['evidence'];o=e['orientation']
                    a,b=o['first'],o['second']
                    both_supported=a['status']=='SUPPORTED' and b['status']=='SUPPORTED'
                    pattern=1 if both_supported and a['ci_low']>0 and b['ci_high']<0 else -1 if both_supported and b['ci_low']>0 and a['ci_high']<0 else 0
                    good=pattern!=0
                    for ref in ['O1','STATE_ONLY']:
                        c=e['contrasts'][r['method']+'__vs__'+ref]
                        good &= all(c[k]['ci_low'] is not None and c[k]['ci_low']>0 for k in ['TS_AUC','logloss_gain'])
                    if good:patterns.append(pattern)
                if max(patterns.count(1),patterns.count(-1))>=2:
                    selected.append(dict(feature=j,pair_index=pair_index,pair=list(pair),pattern=1 if patterns.count(1)>=2 else -1,
                        status='TRAIN_PROVISIONAL_NOT_OUTER_CERTIFIED'))
                    break
            if len(selected)>=6: break
        mode_record=dict(mode=mode,candidates=candidates,provisional_rules=selected,records=evidence,
            scope=parent.provenance(),candidate_freeze_sha256=digest(BASE/'candidates'/parent.name/'FROZEN.json'))
        atomic(root/mode/'SELECTION_FROZEN.json',mode_record)
        modes.append(dict(mode=mode,features=[r['feature'] for r in selected],sha256=digest(root/mode/'SELECTION_FROZEN.json')))
    assert not violations
    atomic(root/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),scope=parent.provenance(),modes=modes,
        label_access=labels.accesses,outer_scores_released=False,violations=violations,
        implementation_sha256=digest(BASE/'ORIENTATION_SELECTION_IMPLEMENTATION_LOCK_NEXT9.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,required=True,choices=range(5));p.add_argument('--inner',type=int,choices=range(3))
    a=p.parse_args();main(a.fold,a.inner)
