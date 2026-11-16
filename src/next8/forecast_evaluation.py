"""Question A heldout movement and state comparisons; no break labels."""
import numpy as np
from src.next8.common import BASE,read_json,digest
from src.next8.features import UNKNOWN
from src.next8.movement import movement_target
from src.next8.statistics import regression_metrics,classification_metrics,multiclass_losses
from src.next8.diagnostics import original_groups,time_H_domains,loss_evidence


def load_forecasts(cache,scope,j,h,release):
    p=BASE/'forecasts'/scope.name/f'F{j:03d}_h{h}';done=read_json(p/'DONE.json')
    tp=BASE/'transitions'/scope.name/p.name;td=read_json(tp/'DONE.json')
    for path,record in ((p,done),(tp,td)):
        assert record['scope']==scope.provenance()
        f=path/'PREDICTIONS.npz';assert digest(f)==record['files_sha256']['PREDICTIONS.npz']==release['files_sha256'][f.relative_to(BASE).as_posix()]
    with np.load(p/'PREDICTIONS.npz',allow_pickle=False) as a:pred={k:a[k].copy() for k in a.files}
    with np.load(tp/'PREDICTIONS.npz',allow_pickle=False) as a:
        np.testing.assert_array_equal(a['global_row'],pred['global_row'])
        pred.update({k:a[k].copy() for k in a.files if k not in ('global_row','dataset_id')})
    rows=pred['global_row'];scope.admit_validation(np.unique(pred['dataset_id']))
    np.testing.assert_array_equal(cache.meta['dataset_id'][rows],pred['dataset_id'])
    usable=cache.forecast_rows(rows,h);at=np.searchsorted(rows,usable)
    target=movement_target(cache,usable,j,h)
    outputs={k:v[at] for k,v in pred.items() if k not in ('global_row','dataset_id')}
    for method,donor in done['method_aliases'].items():
        for kind in ('delta','movement','reversal','state'):outputs[method+'_'+kind]=outputs[donor+'_'+kind]
    return usable,target,outputs,done,td,len(rows)


def evaluate_forecast(cache,scope,j,h,release):
    rows,target,p,done,transition,issued=load_forecasts(cache,scope,j,h,release)
    groups=original_groups(cache,rows);metrics=[];comparisons=[];group_arrays={};counter=0
    base={'scope':scope.name,'feature':j,'horizon':h,'parents':done['parents']}
    methods=list(done['method_aliases'])
    definitions={'D0_delta':('delta','D0')}
    for method in methods:
        for target_name in ('delta','movement','reversal','state'):definitions[method+'_'+target_name]=(target_name,method)
    for key,kind,method in (
        ('D0_movement_majority','movement','D0_MAJORITY'),('D0_movement_persistence','movement','D0_PERSISTENCE'),
        ('D0_reversal','reversal','D0_MAJORITY'),('D0_state_majority','state','T0_MAJORITY'),
        ('D0_state_persistence','state','T0_PERSISTENCE'),('T1_state','state','T1'),('T2_state','state','T2'),
        ('T3_state','state','T3'),('T4_state','state','T4')):definitions[key]=(kind,method)
    for family,domain,keep in time_H_domains(cache,rows):
        if not keep.any():
            metrics.append({**base,'domain':domain,'status':'INSUFFICIENT_SUPPORT','rows':0});continue
        for key,(kind,method) in definitions.items():
            masks=[('ALL',keep)] if kind!='state' else [('KNOWN_H',keep&(target['state']<UNKNOWN)),('ALL_FOUR_STATES',keep)]
            for state_domain,mask in masks:
                record={**base,'domain_family':family,'domain':domain,'state_domain':state_domain,'target':kind,'method':method,
                    'alias':done['method_aliases'].get(method,method),'T4_identical_parent_contrast':transition['T4_alias_T3_identical_parents'] if method=='T4' else None}
                if not mask.any():metrics.append({**record,'status':'INSUFFICIENT_SUPPORT','rows':0});continue
                function=regression_metrics if kind=='delta' else classification_metrics
                r=function(target[kind][mask],p[key][mask],groups[mask])
                metrics.append({**record,**r,'status':'SUPPORTED' if r['independent_series']>=64 else 'INSUFFICIENT_SUPPORT'})
        pairs=[]
        # Matched ablations use the same targets and rows, never best-horizon selection.
        chain=[('D0','D1'),('D1','D2'),('D2','D3'),('D3','D4'),('D5','D5_CROSS'),('D1','D5_CROSS')]
        chain += [(f'D5_REMOVE_F{parent:03d}','D5_CROSS') for parent in done['parents']]
        if family!='ALL':chain=[pair for pair in chain if pair[1]=='D5_CROSS']
        for ref,cand in chain:
            pairs.append((ref+'_delta',cand+'_delta','delta','MSE',keep))
            if family=='ALL':
                for kind in ('movement','reversal'):
                    reference=(('D0_movement_majority' if kind=='movement' else 'D0_reversal') if ref=='D0' else ref+'_'+kind)
                    pairs.append((reference,cand+'_'+kind,kind,'LOGLOSS',keep))
        if family=='ALL':
            pairs.append(('D0_movement_persistence','D1_movement','movement','LOGLOSS',keep))
            for ref in ('D0_state_persistence','D0_state_majority','T1_state','T2_state','T4_state'):
                pairs.append((ref,'T3_state','state','LOGLOSS',keep&(target['state']<UNKNOWN)))
        for ref,cand,kind,metric,mask in pairs:
            if metric=='MSE':a=(target[kind][mask]-p[ref][mask])**2;b=(target[kind][mask]-p[cand][mask])**2
            else:a=multiclass_losses(target[kind][mask],p[ref][mask]);b=multiclass_losses(target[kind][mask],p[cand][mask])
            evidence,keys,gain=loss_evidence(groups[mask],a,b);key=f'C{counter:04d}';counter+=1
            group_arrays[key+'_group']=keys;group_arrays[key+'_gain']=gain
            comparisons.append({**base,'domain_family':family,'domain':domain,'reference':ref,'candidate':cand,
                'target':kind,'metric':metric,'state_domain':'KNOWN_H' if kind=='state' else 'ALL',
                'informative_random_contrast':not transition['T4_alias_T3_identical_parents'] if ref=='T4_state' else True,
                'group_array_key':key,**evidence})
    return {'status':'COMPLETE',**base,'issued_forecasts':issued,'available_future_targets':len(rows),
        'unavailable_suffix_forecasts_retained_but_not_scored':issued-len(rows),'methods':methods,
        'no_break_labels_used':True,'unknown_H_fraction':float(np.mean(target['state']==UNKNOWN)),
        'current_sign_reversal_persistence_point_accuracy':float(np.mean(target['reversal']==0)),
        'metrics':metrics,'comparisons':comparisons,'T4_identical_parent_contrast':transition['T4_alias_T3_identical_parents'],
        'training_scope':done['scope'],'early_training_cutoff':done['early_training_cutoff'],
        'A_is_not_B_or_C':True},group_arrays
