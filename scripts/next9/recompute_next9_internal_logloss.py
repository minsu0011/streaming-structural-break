"""Recompute only invalid losses from frozen predictions; preserve rank evidence."""
from pathlib import Path
import sys
import argparse
import copy
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now
from src.next9.boundary import install_workspace_boundary


def main(fold,inner):
    violations=install_workspace_boundary()
    import numpy as np
    from src.next9.data import FeatureBank,outer_scope,inner_scope,selection_splits
    from src.next9.labels import ScopedLabels
    from src.next9.orientation_contract import CommonRange
    from src.next9.inference import weights_for,interval
    from src.next9.logloss_signed import binary_loss
    from src.next9.movement_screen import grouped_means
    bank=FeatureBank();labels=ScopedLabels(bank)
    parent=outer_scope(bank,fold) if inner is None else inner_scope(bank,fold,inner)
    scopes=selection_splits(bank,fold,inner)
    source=BASE/'orientation_selection'/parent.name
    root=BASE/'orientation_logloss_corrected'/parent.name
    source_done=read_json(source/'DONE.json')
    with np.load(BASE/'state_cache/INDEX.npz') as z:index=z['global_row']
    total=0;recomputed=0;modes=[]
    for mode in ['NORMAL','EARLY']:
        selection=read_json(source/mode/'SELECTION_FROZEN.json')
        member=next(r for r in source_done['modes'] if r['mode']==mode)
        assert digest(source/mode/'SELECTION_FROZEN.json')==member['sha256']
        output=copy.deepcopy(selection);output['records']=[]
        for j in selection['candidates']:
            with np.load(BASE/'state_cache'/f'F{j:03d}'/'STATES.npz') as z:
                states={k:z[k] for k in ['S1_DIRECTION','S2_DIRECTION_RUN','S3_DIRECTION_SLOPE','S4_REVERSAL']}
            for split,scope in enumerate(scopes):
                directory=source/mode/f'F{j:03d}'/f'panel_{split}'
                destination=root/mode/f'F{j:03d}'/f'panel_{split}'/'CORRECTED.json'
                if destination.exists():
                    records=read_json(destination)['records'];output['records'].extend(records)
                    total+=len(records);recomputed+=sum(r['status']=='COMPLETE' for r in records);continue
                with activity(f'{parent.name} {mode} F{j:03d} panel{split} signed loss reconstruction from frozen predictions',
                              'LOGLOSS_NUMERICAL_REPAIR','B9_NORMALIZATION_STATE_AVAILABILITY'):
                    done=read_json(directory/'DONE.json')
                    assert digest(directory/'PREDICTIONS.npz')==done['files_sha256']['PREDICTIONS.npz']
                    with np.load(directory/'PREDICTIONS.npz') as z:
                        rows=z['global_row'];predictions={k:z[k] for k in z.files if k!='global_row'}
                    loc=np.searchsorted(index,rows);np.testing.assert_array_equal(index[loc],rows)
                    y=labels.get(rows,scope,'INTERNAL_VALIDATION',early=mode=='EARLY')
                    current=bank.current(rows,[j])[:,0]
                    groups=np.asarray([bank.guard.split.groups[int(s)] for s in bank.row_ids[rows]],dtype=np.int32)
                    records=copy.deepcopy([r for r in selection['records'] if r['feature']==j and r['split']==split])
                    for r in records:
                        total+=1
                        if r['status']!='COMPLETE':continue
                        recomputed+=1
                        domain=CommonRange(**r['domain']);keep=domain.mask(current,states[r['pair'][0]][loc])
                        pred={k:v[keep] for k,v in predictions.items() if not k.startswith('STATE_ONLY_')}
                        pred['STATE_ONLY']=predictions[r['matched_state_only']][keep]
                        names=list(pred);unique,means=grouped_means(np.column_stack([binary_loss(y[keep],pred[k]) for k in names]),groups[keep])
                        weights,receipt=weights_for(unique,r['evidence']['bootstrap']['partition'])
                        assert receipt==r['evidence']['bootstrap']
                        samples=(weights@means)/weights.sum(axis=1)[:,None]
                        points=means.mean(axis=0);lookup={name:k for k,name in enumerate(names)}
                        original_evidence=copy.deepcopy(r['evidence'])
                        r['evidence']['logloss']={name:float(points[k]) for k,name in enumerate(names)}
                        for contrast,result in r['evidence']['contrasts'].items():
                            candidate,reference=contrast.split('__vs__');a=lookup[candidate];b=lookup[reference]
                            result['logloss_gain']=interval(points[b]-points[a],samples[:,b]-samples[:,a])
                            assert result['TS_AUC']==original_evidence['contrasts'][contrast]['TS_AUC']
                        assert r['evidence']['orientation']==original_evidence['orientation']
                        assert r['evidence']['TS_AUC']==original_evidence['TS_AUC']
                        r['logloss_signed_repair']=dict(original_label_dtype=str(y.dtype),recomputed_float64=True,
                            frozen_prediction_sha256=done['files_sha256']['PREDICTIONS.npz'],same_whole_group_bootstrap=True,
                            rank_and_orientation_evidence_unchanged=True)
                    atomic(destination,dict(status='COMPLETE',timestamp=utc_now(),records=records,
                        original_selection_sha256=digest(source/mode/'SELECTION_FROZEN.json')))
                    output['records'].extend(records)
        selected=[]
        for j in selection['candidates']:
            for pair in range(8):
                patterns=[]
                for r in output['records']:
                    if r['feature']!=j or r['pair_index']!=pair or r['status']!='COMPLETE' or r['difference_Holm_p']>=.05:continue
                    e=r['evidence'];a=e['orientation']['first'];b=e['orientation']['second']
                    if a['status']!='SUPPORTED' or b['status']!='SUPPORTED':continue
                    pattern=1 if a['ci_low']>0 and b['ci_high']<0 else -1 if b['ci_low']>0 and a['ci_high']<0 else 0
                    good=pattern!=0
                    for ref in ['O1','STATE_ONLY']:
                        contrast=e['contrasts'][r['method']+'__vs__'+ref]
                        good &= all(contrast[k]['ci_low'] is not None and contrast[k]['ci_low']>0 for k in ['TS_AUC','logloss_gain'])
                    if good:patterns.append(pattern)
                if max(patterns.count(1),patterns.count(-1))>=2:selected.append((j,pair))
        assert not selected and not selection['provisional_rules'],'Selection changed: stop dependent release'
        output.update(logloss_correction='Authoritative signed float64 losses; original records retained as invalid-loss provenance.',
                      original_selection_sha256=digest(source/mode/'SELECTION_FROZEN.json'),selection_unchanged=True)
        atomic(root/mode/'SELECTION_FROZEN.json',output)
        modes.append(dict(mode=mode,sha256=digest(root/mode/'SELECTION_FROZEN.json')))
    assert not violations
    atomic(root/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),scope=parent.provenance(),modes=modes,
        pair_panels=total,recomputed_complete_panels=recomputed,selection_unchanged=True,new_fits=0,
        no_outer_evidence_used=True,original_done_sha256=digest(source/'DONE.json'),label_access=labels.accesses,violations=violations))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,required=True,choices=range(5));p.add_argument('--inner',type=int,choices=range(3))
    a=p.parse_args();main(a.fold,a.inner)
