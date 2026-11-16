"""Observable forecast designs and fixed mechanical-controlled parent selection.

Targets are future feature values. They are never returned by design builders.
Graph selection is a training-only operation and uses no breakOccurred labels.
"""
import hashlib
from pathlib import Path
import numpy as np
import joblib
from src.next8.common import BASE,atomic,read_json,digest,utc_now,activity
from src.next8.features import DETAILED,NAMES,HORIZONS,mechanical_controls,relation_type,categorize,movement_direction
from src.next8.data import FitScope
from src.next8.learning import Design,fit_ridge,fit_logistic,fit_tree,select_training_only
from src.next8.statistics import (series_weights,cluster_difference,holm,regression_metrics,
    classification_metrics,multiclass_losses)
from src.next8.screening import sampled_rows,selection_splits,procedure_scope
from src.next8.protocol import TIME_BUCKETS,H_BUCKETS


def history_value(cache,rows,columns,lag):
    """The last16 observed/H values, with an explicit series boundary check."""
    if not 1<=lag<=16:raise ValueError('Fixed16-value duration memory')
    if lag in (1,4,16):return cache.lag(rows,columns,lag)
    rows=np.asarray(rows,dtype=np.int64);cols=np.asarray(columns,dtype=np.int32)
    t=cache.meta['time_online'][rows].astype(np.int64);online=t>=lag
    out=np.empty((len(rows),len(cols)),dtype=np.float32)
    past=rows[online]-lag
    if not np.array_equal(cache.meta['dataset_id'][past],cache.meta['dataset_id'][rows[online]]):
        raise ValueError('Cross-series history rejected')
    out[online]=cache.current(past,cols)
    s=cache.series_index[rows[~online]]
    out[~online]=cache.series['seed_values'][s[:,None],(16+t[~online]-lag)[:,None],cols[None,:]]
    return out


def own_design(cache,rows,j,horizon,context=False):
    rows=np.asarray(rows,dtype=np.int64);now=cache.current(rows,[j])[:,0]
    lags=[cache.lag(rows,[j],k)[:,0] for k in (1,4,16)]
    sign=np.sign(now);active=np.ones(len(rows),dtype=bool);duration=np.ones(len(rows))
    for k in range(1,16):
        active &= np.sign(history_value(cache,rows,[j],k)[:,0])==sign
        duration+=active
    values=[now,*lags,*( (now-v)/k for v,k in zip(lags,(1,4,16)) ),sign,duration,np.abs(now),cache.epsilon(rows,[j],horizon)[:,0]]
    names=[f'F{j:03d}_observed_lag{k}' for k in (0,1,4,16)]+['own_slope1','own_slope4','own_slope16',
           'own_value_sign','own_sign_duration_capped16','own_abs_F','H_own_flat_epsilon']
    if context:
        c=cache.context(rows);values.extend(c.T);names.extend([f'H_or_observed_time_context_{k}' for k in range(c.shape[1])])
    return Design(np.column_stack(values).astype(np.float32),tuple(names),cache.meta['dataset_id'][rows])


def augment(cache,rows,design,columns,*,current=False,prefix='parent'):
    unique=[int(k) for k in dict.fromkeys(columns)]
    if not unique:return design
    pieces=[design.values];names=list(design.names)
    for lag in ((0,1,4) if current else (1,4,16)):
        f=cache.current(rows,unique) if lag==0 else cache.lag(rows,unique,lag)
        for k,j in enumerate(unique):
            # Semantic coordinate identity is independent of an ablation's
            # block name: duplicate copies must not weaken ridge regularization.
            name=f'F{j:03d}_observed_lag{lag}'
            if name not in names:pieces.append(f[:,k:k+1]);names.append(name)
    return Design(np.column_stack(pieces).astype(np.float32),tuple(names),design.ids)


def mechanical_design(cache,rows,j,horizon,parents=()):
    own=own_design(cache,rows,j,horizon,context=True)
    result=augment(cache,rows,own,mechanical_controls(j),current=True,prefix='mechanical')
    return augment(cache,rows,result,[p for p in parents if p!=j],prefix='cross')


def forecast_delta(model,design,current):
    """Respect the known [-12,12] feature range, identically for all models."""
    current=np.asarray(current)
    return np.clip(current+model.predict(design),-12.,12.)-current


def cross_eligible(parent,destination):
    return parent!=destination and parent not in mechanical_controls(destination) and relation_type(parent,destination)!='MECHANICAL_DERIVED'


def movement_target(cache,rows,j,horizon):
    """Only call after forecast_rows admission; callers cannot request a suffix."""
    rows=np.asarray(rows,dtype=np.int64)
    if not np.array_equal(rows,cache.forecast_rows(rows,horizon)):
        raise ValueError('Unavailable or cross-series auxiliary target')
    current=cache.current(rows,[j])[:,0];following=cache.current(rows+horizon,[j])[:,0]
    epsilon=cache.epsilon(rows,[j],horizon)[:,0]
    s=cache.series_index[rows]
    low=cache.series['low'][s,j];high=cache.series['high'][s,j];valid=cache.series['valid'][s,j]
    delta=following-current
    return {'future_F':following,'delta':delta,'movement':movement_direction(delta,epsilon),
            'state':categorize(following,low,high,valid),
            'reversal':(np.sign(following)*np.sign(current)<0).astype(np.int8),
            'available':np.ones(len(rows),dtype=bool)}


def _domain_gains(cache,rows,reference_loss,candidate_loss):
    result=[];ids=cache.meta['dataset_id'][rows]
    t=cache.meta['time_online'][rows];hlen=cache.series['H_length'][cache.series_index[rows]]
    for kind,values,buckets in (('time',t,TIME_BUCKETS),('H_length',hlen,H_BUCKETS)):
        for low,high in buckets:
            keep=(values>=low)&(True if high is None else values<=high)
            if len(np.unique(ids[keep]))<64:continue
            _,inverse,count=np.unique(ids[keep],return_inverse=True,return_counts=True)
            gain=np.bincount(inverse,weights=reference_loss[keep]-candidate_loss[keep])/count
            result.append({'domain_type':kind,'domain':f'{low}:{high}','series':len(count),
                           'gain':float(gain.mean()),'status':'SUPPORTED'})
    return result


def parent_selection_split(cache,scope,directory,cutoff=None):
    directory.mkdir(parents=True,exist_ok=True)
    select_training_only({'purpose':'fixed eligible parent testing'},scope,scope.training_ids,'graph')
    tr_all=sampled_rows(cache,scope.training_ids,cutoff);va_all=sampled_rows(cache,scope.validation_ids,cutoff)
    records=[]
    for j in DETAILED:
        for horizon in HORIZONS:
            path=directory/f'F{j:03d}_h{horizon}.json'
            if path.exists():
                record=read_json(path);assert record['scope']==scope.provenance();records.append(record);continue
            with activity(f'Mechanical parent screen {scope.name} F{j:03d} h{horizon}', 'MOVEMENT_PARENT_SELECTION','B3_SELF_HISTORY_SUFFICIENCY'):
                tr=cache.forecast_rows(tr_all,horizon,cutoff);va=cache.forecast_rows(va_all,horizon,cutoff)
                scope.admit_training(np.unique(cache.meta['dataset_id'][tr]));scope.admit_validation(np.unique(cache.meta['dataset_id'][va]))
                train=mechanical_design(cache,tr,j,horizon);valid=mechanical_design(cache,va,j,horizon)
                yt=movement_target(cache,tr,j,horizon)['delta'];yv=movement_target(cache,va,j,horizon)['delta']
                weight=series_weights(train.ids)
                baseline=fit_ridge(train,yt,scope,weight)
                current=cache.current(va,[j])[:,0]
                prediction=forecast_delta(baseline,valid,current)
                loss=(prediction-yv)**2
                candidates=[]
                for parent in DETAILED:
                    edge=relation_type(parent,j)
                    if not cross_eligible(parent,j):
                        candidates.append({'parent':parent,'edge_type':edge,'status':'MECHANICAL_OR_SELF_NOT_CROSS',
                            'incremental_test':'Primitive already in strong own/mechanical controls; extra duplicate coordinates or longer history do not count as a new factor','common_raw':True})
                        continue
                    td=augment(cache,tr,train,[parent],prefix='cross');vd=augment(cache,va,valid,[parent],prefix='cross')
                    model=fit_ridge(td,yt,scope,weight);p=forecast_delta(model,vd,current);candidate_loss=(p-yv)**2
                    evidence=cluster_difference(valid.ids,loss,candidate_loss)
                    domains=_domain_gains(cache,va,loss,candidate_loss)
                    candidates.append({'parent':parent,'edge_type':'CROSS_PREDICTIVE_CANDIDATE',
                        'status':'TESTED','common_raw':True,'lags':[1,4,16],**evidence,
                        'domain_results':domains,'baseline_MSE':float(np.average(loss,weights=series_weights(valid.ids))),
                        'candidate_MSE':float(np.average(candidate_loss,weights=series_weights(valid.ids)))})
                tested=[c for c in candidates if c['status'] in ('SUPPORTED','INSUFFICIENT_SUPPORT') and 'p_normal_cluster' in c]
                # cluster_difference supplies support status; never confuse this
                # with discovery certification or an independent hidden test.
                corrections=holm([c['p_normal_cluster'] for c in tested])
                for c,p in zip(tested,corrections):c['Holm_p_within_feature_horizon']=float(p)
                record={'status':'COMPLETE','timestamp':utc_now(),'feature':j,'horizon':horizon,
                    'scope':scope.provenance(),'training_rows':len(tr),'validation_rows':len(va),
                    'mechanical_feature_columns':list(mechanical_controls(j)),'baseline':regression_metrics(yv,prediction,valid.ids),
                    'candidates':candidates,'breakOccurred_labels_used':False,
                    'source_feature_future_is_supervision_only':True,
                    'strong_control_limit':'Complete named same-primitive bank plus raw/robust innovation, raw energy and forecast current/lag1/lag4; not an exhaustive latent AR/filter-state sufficient statistic.'}
                atomic(path,record);records.append(record)
    # One family across all12 features x3 horizons x tested parents, not only
    # a favorable selected subset. Bonferroni/Holm handles recorded tests.
    tests=[(r['feature'],r['horizon'],c['parent'],c['p_normal_cluster']) for r in records for c in r['candidates'] if 'p_normal_cluster' in c]
    adjusted=holm([t[-1] for t in tests])
    correction=[{'feature':j,'horizon':h,'parent':p,'raw_p':pv,'Holm_p':float(a)} for (j,h,p,pv),a in zip(tests,adjusted)]
    atomic(directory/'MULTIPLICITY.json',{'family':'All12 detailed targets x3 horizons x all nonmechanical eligible parents in this training split','tests':correction})
    atomic(directory/'DONE.json',{'status':'COMPLETE','timestamp':utc_now(),'scope':scope.provenance(),
        'records_sha256':{f'F{r["feature"]:03d}_h{r["horizon"]}.json':digest(directory/f'F{r["feature"]:03d}_h{r["horizon"]}.json') for r in records},
        'breakOccurred_labels_used':False,'outer_validation_excluded':True})
    return records,correction


def select_parent_graph(panels,corrections,scope):
    candidates=[]
    for j in DETAILED:
        for parent in DETAILED:
            if not cross_eligible(parent,j):continue
            horizon_evidence=[]
            for h in HORIZONS:
                good=[];gains=[];domains=[]
                for k,(panel,adjustment) in enumerate(zip(panels,corrections)):
                    r=next(r for r in panel if r['feature']==j and r['horizon']==h)
                    c=next(c for c in r['candidates'] if c['parent']==parent)
                    a=next(a for a in adjustment if a['feature']==j and a['horizon']==h and a['parent']==parent)
                    gains.append(c['gain']/max(c['baseline_MSE'],1e-8))
                    if c['status']=='SUPPORTED' and c['ci_low'] is not None and c['ci_low']>0 and a['Holm_p']<.05:
                        good.append(k);domains.append({(d['domain_type'],d['domain']) for d in c['domain_results'] if d['gain']>0})
                common=set.intersection(*domains) if domains else set()
                supported=len(good)>=2 and all(sum(k==kind for k,_ in common)>=2 for kind in ('time','H_length'))
                horizon_evidence.append({'horizon':h,'supported':supported,'supporting_inner_splits':good,
                    'mean_relative_MSE_gain':float(np.mean(gains)),
                    'common_domains':[list(v) for v in sorted(common)]})
            supported=[h for h in horizon_evidence if h['supported']]
            candidates.append({'source':parent,'destination':j,'lags':[1,4,16],
                'edge_type':'CROSS_PREDICTIVE' if supported else 'UNRESOLVED_CROSS_PREDICTIVE',
                'common_raw':True,'causal':False,'independent_confirmation':False,
                'horizons':horizon_evidence,'supported':bool(supported),
                'selection_score':max((h['mean_relative_MSE_gain'] for h in supported),default=0.),
                'selection_basis':'Training-only repeated inner MSE improvement beyond D5 and whole-family Holm; final matched removal and outer validation still required'})
    selected=[];counts={j:0 for j in DETAILED}
    for c in sorted([c for c in candidates if c['supported']],key=lambda c:(-c['selection_score'],c['destination'],c['source'])):
        if len(selected)>=12:break
        if counts[c['destination']]>=2:continue
        selected.append(c);counts[c['destination']]+=1
    return {'status':'FROZEN','timestamp':utc_now(),'scope':scope.provenance(),'edges':selected,
        'certification':'INNER_SELECTION_ONLY_AWAIT_OUTER_AND_MATCHED_ABLATION',
        'all_candidates':candidates,'max_edges':12,'max_parents':2,
        'target_features':list(DETAILED),'parent_pool':list(DETAILED),
        'stable_block_control':'D3 evaluates the separately training-certified stable feature block. Parent certification is limited to this fixed12-feature pool; a D3 block gain without a supported individual parent remains unresolved, not NO_CROSS_FACTOR for all possible features.',
        'breakOccurred_labels_used_for_parent_selection':False,'outer_or_meta_validation_used':False,
        'no_supported_edges_status':'NO_CROSS_FACTOR_WITHIN_FIXED_PARENT_MENU' if not selected else None}
