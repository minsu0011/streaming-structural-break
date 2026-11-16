"""Once-frozen prediction evaluation; no fitting or candidate selection here."""
import hashlib
import numpy as np
from src.next8.common import BASE,read_json,digest
from src.next8.data import baseline_oof
from src.next8.integration import fixed_blend
from src.next8.h_context import HStateSpec,CONTEXT_MENU
from src.next8.protocol import TIME_BUCKETS,H_BUCKETS
from src.next8.paired_bootstrap import evaluate_partition,PreparedRank

METHODS=('M0','M1','M2','M3','M4','M5','M6','BL95_M3_05_CONTROL','BL95_M4_05_CONTROL','BL95_M2_05_CONTROL')


def frozen_predictions(cache,scope,fold,*,early=False):
    p=BASE/'models'/scope.name;done=read_json(p/'DONE.json')
    assert done['scope']==scope.provenance()
    assert digest(p/'PREDICTIONS.npz')==done['files_sha256']['PREDICTIONS.npz']
    with np.load(p/'PREDICTIONS.npz',allow_pickle=False) as a:
        rows=a['global_row'].copy();ids=a['dataset_id'].copy();predictions={m:a[m].copy() for m in ('M1','M2','M3','M4','M5')}
    scope.admit_validation(np.unique(ids));np.testing.assert_array_equal(cache.meta['dataset_id'][rows],ids)
    if np.any(np.diff(rows)<=0):raise ValueError('Ordered unique complete prediction rows required')
    if early:
        bp=BASE/'early_baseline'/scope.name;bd=read_json(bp/'DONE.json')
        assert bd['scope']==scope.provenance() and digest(bp/'BL80_PREDICTIONS.npz')==bd['prediction_sha256']
        with np.load(bp/'BL80_PREDICTIONS.npz',allow_pickle=False) as a:baseline={k:a[k].copy() for k in a.files}
        ip=BASE/'early_integration'/scope.name/'DONE.json'
        assert (cache.meta['time_online'][rows]>=400).all()
    else:
        baseline=baseline_oof(fold,data=cache.data);ip=BASE/'integration'/scope.name/'DONE.json'
    np.testing.assert_array_equal(baseline['global_row'],rows)
    predictions['M0']=baseline['prediction'];integration=read_json(ip)
    if integration['scope']!=scope.provenance():raise ValueError('Wrong integration eligibility scope')
    eligible=integration['eligible']
    predictions['M6']=fixed_blend(predictions['M0'],predictions['M5']) if eligible else predictions['M0']
    for method in ('M3','M4','M2'):
        predictions[f'BL95_{method}_05_CONTROL']=fixed_blend(predictions['M0'],predictions[method]) if eligible else predictions['M0']
    return rows,{name:predictions[name] for name in METHODS},{'scope':scope.provenance(),
        'prediction_sha256':digest(p/'PREDICTIONS.npz'),'integration_sha256':digest(ip),'integration_eligible':eligible,
        'integration_status':integration['M6_status'],'model_aliases':done['model_aliases'],
        'same_BL_controls_use_exact_same_eligibility_mask':True,'early_reference':early}


def domains(cache,scope,rows,specs):
    rows=np.asarray(rows,dtype=np.int64);t=cache.meta['time_online'][rows]
    hlength=np.diff(cache.data.series['historical_offsets'])[cache.series_index[rows]]
    output=[('ALL','ALL',np.ones(len(rows),dtype=bool),{})]
    for family,buckets,values in (('TIME',TIME_BUCKETS,t),('H_LENGTH',H_BUCKETS,hlength)):
        for k,(lo,hi) in enumerate(buckets):
            output.append((family,f'{family}_{k}',(values>=lo)&(True if hi is None else values<=hi),{'lower':lo,'upper':hi}))
    for kind in CONTEXT_MENU:
        spec=HStateSpec(**specs[kind])
        if spec.training['training_ids_sha256']!=scope.provenance()['training_ids_sha256']:
            raise ValueError('Domain map must be frozen using this exact TRAIN membership')
        state=spec.states_from_context(cache.series['context'][cache.series_index[rows]])
        for s in range(4):output.append((kind,f'{kind}_{s}',state==s,{'state':s,'training_map':vars(spec),'constant_within_series':True}))
    return output


def group_support(y,groups):
    y=np.asarray(y);groups=np.asarray(groups)
    counts={'rows':len(y),'groups':len(np.unique(groups)),
        'positive_groups':len(np.unique(groups[y==1])),'negative_groups':len(np.unique(groups[y==0]))}
    return {**counts,'major_domain_supported':min(counts['positive_groups'],counts['negative_groups'])>=64}


def exact_unique_partition(y,predictions,t,groups,*,partition,draws=2000,require_support=True):
    """Deduplicate bitwise-equal float64 probabilities, retaining every name."""
    unique={};aliases={};seen={}
    for name,p in predictions.items():
        value=np.asarray(p,dtype=np.float64)
        sha=hashlib.sha256(value.tobytes()).hexdigest()
        if sha in seen:
            donor=seen[sha];np.testing.assert_array_equal(unique[donor],value);aliases[name]=donor
        else:seen[sha]=name;aliases[name]=name;unique[name]=value
    support=group_support(y,groups)
    if len(y) and (support['major_domain_supported'] or not require_support):
        point,sample,receipt=evaluate_partition(y,unique,t,groups,partition=partition,draws=draws)
        receipt['status']='SUPPORTED' if all(p['pairs']>0 for p in point.values()) else 'INSUFFICIENT_SUPPORT'
    else:
        point={};sample={}
        for name,p in unique.items():
            rank=PreparedRank.prepare(y,p,t,groups) if len(y) else None
            point[name]={'TS_AUC':rank.point_auc if rank is not None and rank.point_pairs else None,'pairs':rank.point_pairs if rank else 0}
            sample[name]=np.full(draws,np.nan)
        receipt={'status':'INSUFFICIENT_SUPPORT','draws':draws,'reason':'Fewer than64 original groups in either class; point estimate descriptive only','zero_pair_or_sparse_draws_not_fabricated':True}
    return {n:point[d] for n,d in aliases.items()},{n:sample[d] for n,d in aliases.items()}, {
        **receipt,'support':support,'aliases':aliases,'exact_prediction_duplicates_evaluated_once':True,
        'methods_all_retained':list(predictions),'unique_prediction_arrays':len(unique)}


def simultaneous_envelope(points,samples,reference='M0',methods=('M1','M2','M3','M4','M5','M6')):
    if not points or len(points)!=len(samples):raise ValueError('Matched folds required')
    if any(p[m]['TS_AUC'] is None for p in points for m in (reference,*methods)):
        return {'status':'INSUFFICIENT_SUPPORT','reason':'Zero-pair partition'}
    estimate=np.array([np.mean([p[m]['TS_AUC']-p[reference]['TS_AUC'] for p in points]) for m in methods])
    boot=np.stack([np.mean(np.stack([s[m]-s[reference] for s in samples]),axis=0) for m in methods])
    valid=np.isfinite(boot).all(axis=0)
    if valid.sum()<.95*boot.shape[1]:return {'status':'INSUFFICIENT_SUPPORT','valid_draws':int(valid.sum())}
    radius=float(np.quantile(np.max(np.abs(boot[:,valid]-estimate[:,None]),axis=0),.95))
    return {'status':'SUPPORTED','reference':reference,'methods':list(methods),'draws':boot.shape[1],
        'valid_draws':int(valid.sum()),'maximum_absolute_centered_bootstrap_radius':radius,
        'intervals':{m:{'mean_delta':float(d),'simultaneous_ci_low':float(d-radius),'simultaneous_ci_high':float(d+radius)} for m,d in zip(methods,estimate)},
        'scope':'Finite frozen six-model family, fixed fitted models; does not correct historical adaptive DEV use',
        'method':'95th percentile of max absolute centered paired equal-fold bootstrap errors'}
