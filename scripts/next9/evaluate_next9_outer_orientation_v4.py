"""Frozen current-F common-range tests on original excluded outer groups."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,atomic,read_json,digest,activity,utc_now
from src.next9.boundary import install_workspace_boundary


def main(fold):
    violations=install_workspace_boundary()
    lock=read_json(BASE/'OUTER_EVALUATION_IMPLEMENTATION_LOCK_V4_NEXT9.json')
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    global_path=BASE/'GLOBAL_PREDICTIONS_FROZEN_NEXT9.json'
    global_freeze=read_json(global_path);assert global_freeze['status']=='FROZEN'
    release=read_json(BASE/'OUTER_RELEASE_NEXT9.json')
    assert release['global_prediction_freeze_sha256']==digest(global_path)
    import numpy as np
    from src.next9.data import FeatureBank,outer_scope
    from src.next9.labels import ScopedLabels
    from src.next9.domains import masks
    from src.next9.orientation_contract import CommonRange,PAIR_MENU,support,sufficient
    from src.next9.inference_signed_v4 import compare
    from src.next9.eligibility import holm
    from src.next9.certification import opposite_pattern,orientation_pass,tabular_comparison
    bank=FeatureBank();scope=outer_scope(bank,fold);labels=ScopedLabels(bank)
    root=BASE/'outer_orientation_evaluation'/scope.name
    if (root/'DONE.json').exists():return
    closure=BASE/'outer_orientation_refits'/scope.name/'DONE.json'
    assert digest(closure)==global_freeze['source_receipts_sha256'][closure.relative_to(BASE).as_posix()]
    scope_record=read_json(closure);mode_records=[]
    for mode in ['NORMAL','EARLY']:
        source=closure.parent/mode
        member=next(r for r in scope_record['records'] if r['mode']==mode)
        assert digest(source/'DONE.json')==member['sha256']
        manifest=read_json(source/'DONE.json');all_records=[]
        for feature in manifest['features']:
            j=feature['feature'];directory=source/f'F{j:03d}'
            assert digest(directory/'DONE.json')==feature['sha256']
            done=read_json(directory/'DONE.json')
            assert digest(directory/'FROZEN.json')==done['frozen_sha256']
            frozen=read_json(directory/'FROZEN.json')
            assert frozen['scope']==scope.provenance()
            assert digest(directory/'PREDICTIONS.npz')==frozen['files_sha256']['PREDICTIONS.npz']
            with np.load(directory/'PREDICTIONS.npz') as z:
                rows=z['global_row'];current=z['current_F'];states={p[0]:z[p[0]] for p in PAIR_MENU}
                predictions={k:z[k] for k in ['O0','O1','O2_LEVEL','O3_MOVE','O4_MOVE_RUN','O5_MOVE_REV',
                    'O6_TABULAR','STATE_ONLY','STATE_ONLY_MOVE','STATE_ONLY_RUN','F_ONLY']}
            y=labels.get(rows,scope,'FROZEN_OUTER_EVALUATION',prediction_receipt=frozen)
            groups=np.asarray([bank.guard.split.groups[int(s)] for s in bank.row_ids[rows]],dtype=np.int32)
            rule=frozen['provisional_rule']
            for pair_index,pair in enumerate(PAIR_MENU):
                out=root/mode/f'F{j:03d}'/f'pair_{pair_index}'
                if (out/'DONE.json').exists():all_records.extend(read_json(out/'DONE.json')['records']);continue
                with activity(f'{scope.name} {mode} F{j:03d} pair{pair_index} outer common-F orientation and fixed domains',
                              'OUTER_ORIENTATION_EVALUATION','B1_COMMON_CURRENT_RANGE'):
                    domain=CommonRange(**frozen['ranges'][pair_index]);state=states[pair[0]]
                    common=domain.mask(current,state)
                    method='O4_MOVE_RUN' if pair[0]=='S2_DIRECTION_RUN' else 'O5_MOVE_REV' if pair[0]=='S4_REVERSAL' else 'O3_MOVE'
                    control='STATE_ONLY_RUN' if method=='O4_MOVE_RUN' else 'STATE_ONLY' if method=='O5_MOVE_REV' else 'STATE_ONLY_MOVE'
                    records=[]
                    for family,name,domain_mask in masks(bank,rows,frozen['domain_spec']):
                        keep=common&domain_mask
                        stats={str(k):support(groups,y,keep&(state==k)) for k in pair[1:]}
                        # Domain inference is a prospectively gated replication
                        # of this TRAIN-chosen rule, never a search over outer data.
                        infer=name=='ALL' or (rule is not None and rule['pair_index']==pair_index)
                        record=dict(fold=fold,mode=mode,feature=j,pair_index=pair_index,pair=list(pair),method=method,
                            family=family,domain_name=name,domain=domain.record(),validation_support=stats,
                            prediction_freeze_sha256=digest(directory/'FROZEN.json'),provisional_rule=rule,
                            difference_p=1.,matched_state_only=control,inference_preselected=infer,
                            rows=int(keep.sum()),outer_group_count=len(np.unique(groups[keep])))
                        if domain.status!='SUPPORTED' or not all(sufficient(v) for v in stats.values()):
                            record['status']='INSUFFICIENT_COMMON_RANGE'
                        elif not infer:
                            record['status']='SUPPORTED_DIAGNOSTIC_ONLY_NOT_TRAIN_SELECTED'
                        else:
                            pred={k:v[keep] for k,v in predictions.items() if not k.startswith('STATE_ONLY_')}
                            pred['STATE_ONLY']=predictions[control][keep];pred['CURRENT_F_DIRECTION']=current[keep]
                            result=compare(y[keep],pred,bank.times[rows[keep]],groups[keep],
                                f'OUTER_B|{scope.name}|{mode}|F{j}|pair{pair_index}|{name}',direction_states=state[keep],pair=pair[1:])
                            record.update(status='COMPLETE',evidence=result,difference_p=result['orientation']['difference']['p_two_sided'])
                        records.append(record)
                    atomic(out/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),records=records,
                        global_freeze_sha256=digest(global_path),no_fitting_or_selection=True))
                    all_records.extend(records)
        # Each fixed domain has 8*all frozen A candidates, including untested
        # p1 records. Distinct domain families are explicitly not simultaneous.
        for domain_name in sorted(set(r['domain_name'] for r in all_records)):
            selected=[r for r in all_records if r['domain_name']==domain_name]
            key=lambda r:(r['feature'],r['pair_index'],r['domain_name'])
            adjusted=holm({key(r):r['difference_p'] for r in selected})
            for r in selected:
                r['difference_Holm_p']=adjusted[key(r)]
                r['orientation_pattern']=opposite_pattern(r['evidence']) if r['status']=='COMPLETE' else 0
                r['orientation_pass']=orientation_pass(r)
                r['tabular_comparison']=tabular_comparison(r)
        atomic(root/mode/'RESULTS.json',dict(status='COMPLETE',timestamp=utc_now(),scope=scope.provenance(),mode=mode,
            records=all_records,multiplicity='Each prespecified domain: every frozen A feature x8 pairs, untested p1. Domain families reported separately; no simultaneous family across domains claimed.',
            no_candidate_rule_or_parameter_changes=True))
        mode_records.append(dict(mode=mode,sha256=digest(root/mode/'RESULTS.json')))
    assert not violations
    atomic(root/'DONE.json',dict(status='COMPLETE',timestamp=utc_now(),scope=scope.provenance(),records=mode_records,
        global_freeze_sha256=digest(global_path),label_access=labels.accesses))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',required=True,type=int,choices=range(5));main(p.parse_args().fold)
