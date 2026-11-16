"""Complete training-scoped feature classification, with no outer-score feedback."""
import hashlib
from pathlib import Path
import numpy as np
import joblib
from src.next8.common import BASE,atomic,read_json,digest,utc_now,activity,ids_hash
from src.next8.features import NAMES,DETAILED,UNKNOWN
from src.next8.protocol import TIME_BUCKETS,H_BUCKETS
from src.next8.data import FitScope,outer_scope,inner_scope,rows_for
from src.next8.learning import (Design,orientation_design,fit_logistic,fit_tree,
    estimate_target_orientation,select_training_only)
from src.next8.statistics import (rank_influence,ranking,binary_losses,cluster_difference,
    logistic_contrast,holm)
from src.next.fitting import weights_from_training


def procedure_scope(cache,fold,meta_inner=None):
    return outer_scope(fold,cache.data) if meta_inner is None else inner_scope(fold,meta_inner,cache.data)


def selection_splits(cache,fold,meta_inner=None):
    parent=procedure_scope(cache,fold,meta_inner)
    if meta_inner is None:return [inner_scope(fold,k,cache.data) for k in range(3)]
    groups={}
    for sid in sorted(parent.training_ids):groups.setdefault(cache.data.guard.split.groups[sid],[]).append(sid)
    def key(g):return hashlib.sha256(f'NEXT8_SUBINNER_V1|{fold}|{meta_inner}|{g}'.encode()).hexdigest()
    assignments={g:k%3 for k,g in enumerate(sorted(groups,key=key))}
    splits=[]
    for k in range(3):
        valid=frozenset(s for g,members in groups.items() if assignments[g]==k for s in members)
        train=parent.training_ids-valid
        excluded=parent.validation_ids|parent.outer_excluded_ids
        cache.data.guard.partition(train,valid|excluded)
        splits.append(FitScope(train,valid,excluded,parent.name+f'_subinner_{k}'))
    return splits


def sampled_rows(cache,ids,cutoff=None):
    r=rows_for(cache.meta,ids);t=cache.meta['time_online'][r]
    keep=t%8==0
    if cutoff is not None:keep &= t<=cutoff
    return r[keep]


def context_for(cache,rows,j):
    return np.column_stack([cache.context(rows),*(cache.lag(rows,[j],k)[:,0] for k in (1,4,16))]).astype(np.float32)


def select_columns(design,keep):
    indices=[k for k,n in enumerate(design.names) if keep(n)]
    return Design(design.values[:,indices],tuple(design.names[k] for k in indices),design.ids)


def _oriented_rank(y,p,t,ids,direction):
    result,_,_=rank_influence(y,p,t,ids)
    if direction<0:
        result['TS_AUC']=1-result['TS_AUC']
        result['ci_low'],result['ci_high']=1-result['ci_high'],1-result['ci_low']
    result['support_status']='SUPPORTED' if min(result['positive_series'],result['negative_series'])>=32 and result['series']>=64 and result['pairs']>0 else 'INSUFFICIENT_SUPPORT'
    return result


def direction_atlas(cache,rows,j,score,direction,mode):
    y=cache.meta['target'][rows];t=cache.meta['time_online'][rows];ids=cache.meta['dataset_id'][rows]
    hlen=cache.series['H_length'][cache.series_index[rows]]
    state=cache.states(rows,[j],previous=True)[:,0]
    scopes=[('all','all',np.ones(len(rows),dtype=bool))]
    for kind,values,buckets in (('time',t,TIME_BUCKETS),('H_length',hlen,H_BUCKETS)):
        for low,high in buckets:scopes.append((kind,f'{low}:{high}',(values>=low)&(True if high is None else values<=high)))
    for k in range(4):scopes.append(('own_pre_state',str(k),state==k))
    output=[]
    for kind,domain,keep in scopes:
        if not np.any(keep):continue
        output.append({'feature_id':f'NX8_F{j:03d}','mode':mode,'domain_type':kind,'domain':domain,
            'direction_from_training':direction,'evaluation':'INNER_HELD_OUT',
            **_oriented_rank(y[keep],score[keep],t[keep],ids[keep],direction)})
    return output


def direction_supported(atlas):
    overall=next(r for r in atlas if r['domain_type']=='all')
    supported=[r for r in atlas if r['support_status']=='SUPPORTED' and r['domain_type']!='all' and not (r['domain_type']=='own_pre_state' and r['domain']=='3')]
    coverage={k:sum(r['domain_type']==k for r in supported) for k in ('time','H_length','own_pre_state')}
    # A direction is not required to have a constant magnitude. The confidence
    # bounds exclude a material reversal (> .03 AUC) in every supported domain.
    # This is a declared practical direction boundary, not exact invariance.
    return bool(overall['support_status']=='SUPPORTED' and overall['ci_low']>.505 and
                all(n>=2 for n in coverage.values()) and all(r['ci_low']>.47 for r in supported)),coverage


def _state_support(ids,y,state):
    return [{'state':k,'series':len(np.unique(ids[state==k])),
             'positive_series':len(np.unique(ids[(state==k)&(y==1)])),
             'negative_series':len(np.unique(ids[(state==k)&(y==0)]))} for k in range(3)]


def common_range(f,state):
    intervals=[]
    for k in range(3):
        values=f[state==k]
        if len(values)<64:return None
        intervals.append(np.quantile(values,[.1,.9]))
    low=max(v[0] for v in intervals);high=min(v[1] for v in intervals)
    return [float(low),float(high)] if high-low>1e-3 else None


def conditional_models(cache,j,scope,training_rows,validation_rows,directory):
    tf=cache.current(training_rows,[j])[:,0];vf=cache.current(validation_rows,[j])[:,0]
    ts=cache.states(training_rows,[j],previous=True)[:,0];vs=cache.states(validation_rows,[j],previous=True)[:,0]
    domain=common_range(tf,ts)
    if domain is None:return {'status':'INSUFFICIENT_SUPPORT','reason':'No shared training q10..q90 feature range','feature':j}
    tr=training_rows[(tf>=domain[0])&(tf<=domain[1])&(ts<UNKNOWN)]
    va=validation_rows[(vf>=domain[0])&(vf<=domain[1])&(vs<UNKNOWN)]
    if not len(tr) or not len(va):return {'status':'INSUFFICIENT_SUPPORT','reason':'Empty common-range train/heldout','feature':j,'domain':domain}
    ys=[cache.meta['target'][r] for r in (tr,va)]
    ids=[cache.meta['dataset_id'][r] for r in (tr,va)]
    states=[cache.states(r,[j],previous=True)[:,0] for r in (tr,va)]
    support=[_state_support(i,y,s) for i,y,s in zip(ids,ys,states)]
    if any(min(row['positive_series'],row['negative_series'])<32 or row['series']<64 for panel in support for row in panel):
        return {'status':'INSUFFICIENT_SUPPORT','reason':'Condition common-range independent class support','feature':j,'domain':domain,'support':support}
    designs=[orientation_design(cache.current(r,[j])[:,0],s,context_for(cache,r,j),i,'O2') for r,s,i in zip((tr,va),states,ids)]
    nulls=[select_columns(d,lambda n:not n.endswith('_times_F')) for d in designs]
    state_only=[select_columns(d,lambda n:not(n.startswith('F_') or n.endswith('_times_F'))) for d in designs]
    w=weights_from_training(ys[0],cache.meta['time_online'][tr],ids[0],'W3')
    models={'O2':fit_logistic(designs[0],ys[0],scope,w),
            'O1_STATE_MAIN':fit_logistic(nulls[0],ys[0],scope,w),
            'STATE_ONLY':fit_logistic(state_only[0],ys[0],scope,w),
            'O4':fit_tree(designs[0],ys[0],scope,w)}
    predictions={name:model.predict_proba(nulls[1] if name=='O1_STATE_MAIN' else state_only[1] if name=='STATE_ONLY' else designs[1])[:,1] for name,model in models.items()}
    losses={name:binary_losses(ys[1],p) for name,p in predictions.items()}
    comparisons={name:cluster_difference(ids[1],losses[name],losses['O2']) for name in ('O1_STATE_MAIN','STATE_ONLY','O4')}
    contrasts=[]
    for a,b in ((0,1),(2,1),(0,2)):
        c=logistic_contrast(models['O2'],designs[0],ys[0],ids[0],w,f'own_state_{a}_times_F',f'own_state_{b}_times_F')
        contrasts.append({'state_a':a,'state_b':b,**c})
    anchors=np.linspace(domain[0],domain[1],3)
    slopes=[{'F':float(a),'derivative':[estimate_target_orientation(models['O2'],a,k) for k in range(3)]} for a in anchors]
    opposing=any(min(row['derivative'])<0<max(row['derivative']) for row in slopes)
    times=cache.meta['time_online'][va];hlen=cache.series['H_length'][cache.series_index[va]]
    held_domains=[]
    for kind,values,buckets in (('time',times,TIME_BUCKETS),('H_length',hlen,H_BUCKETS)):
        for low,high in buckets:
            keep=(values>=low)&(True if high is None else values<=high)
            if len(np.unique(ids[1][keep]))<64 or min(len(np.unique(ids[1][keep&(ys[1]==y)])) for y in (0,1))<32:continue
            held_domains.append({'domain_type':kind,'domain':f'{low}:{high}',
                **cluster_difference(ids[1][keep],losses['O1_STATE_MAIN'][keep],losses['O2'][keep])})
    paths={}
    directory.mkdir(parents=True,exist_ok=True)
    for name,model in models.items():
        p=directory/(name+'.joblib')
        if p.exists():raise FileExistsError(p)
        joblib.dump(model,p);paths[name]={'path':p.relative_to(BASE).as_posix(),'sha256':digest(p)}
    return {'status':'COMPLETE','feature':j,'domain':domain,'support':support,'models':paths,
            'comparisons':comparisons,'contrasts':contrasts,'slopes':slopes,
            'opposing_train_fixed_derivatives_on_common_range':opposing,'heldout_domains':held_domains,
            'derivative_weights_raw_design':dict(zip(models['O2'].names,
                (models['O2'].estimator.coef_[0]/models['O2'].scale).tolist())),
            'converged':all(m.training.get('converged',True) for m in models.values()),
            'provenance':scope.provenance(),'training_rows':len(tr),'validation_rows':len(va),
            'no_validation_labels_used_in_fitting':True,'all_meta_inputs_observable':True}


def screen_split(cache,scope,directory,cutoff=None):
    tr=sampled_rows(cache,scope.training_ids,cutoff);va=sampled_rows(cache,scope.validation_ids,cutoff)
    scope.admit_training(np.unique(cache.meta['dataset_id'][tr]));scope.admit_validation(np.unique(cache.meta['dataset_id'][va]))
    select_training_only({'source':'frozen153-coordinate catalog'},scope,np.unique(cache.meta['dataset_id'][tr]),'feature_classification')
    ty=cache.meta['target'][tr];vy=cache.meta['target'][va]
    tids=cache.meta['dataset_id'][tr];vids=cache.meta['dataset_id'][va]
    w=weights_from_training(ty,cache.meta['time_online'][tr],tids,'W3')
    directory.mkdir(parents=True,exist_ok=True)
    records=[]
    for j,name in enumerate(NAMES):
        path=directory/f'F{j:03d}.json'
        if path.exists():
            record=read_json(path)
            assert record['scope']==scope.provenance()
            records.append(record);continue
        with activity(f'Feature classification {scope.name} F{j:03d}','FEATURE_SCREEN','B9_NESTED_SELECTION'):
            tf=cache.current(tr,[j])[:,0];vf=cache.current(va,[j])[:,0]
            ts=cache.states(tr,[j],previous=True)[:,0];vs=cache.states(va,[j],previous=True)[:,0]
            tc=context_for(cache,tr,j);vc=context_for(cache,va,j)
            d0=[orientation_design(f,s,c,i,'O0') for f,s,c,i in ((tf,ts,tc,tids),(vf,vs,vc,vids))]
            d1=[orientation_design(f,s,c,i,'O1') for f,s,c,i in ((tf,ts,tc,tids),(vf,vs,vc,vids))]
            context=[select_columns(d,lambda n:not n.startswith('F_')) for d in d0]
            models={'O0':fit_logistic(d0[0],ty,scope,w),'O1':fit_logistic(d1[0],ty,scope,w),
                    'CONTEXT_ONLY':fit_logistic(context[0],ty,scope,w)}
            preds={k:m.predict_proba(d0[1] if k=='O0' else d1[1] if k=='O1' else context[1])[:,1] for k,m in models.items()}
            direction=1 if estimate_target_orientation(models['O0'],0.,1)>=0 else -1
            atlas=direction_atlas(cache,va,j,vf,direction,'SIGNED_F')
            nonlinear_atlas=direction_atlas(cache,va,j,preds['O1'],1,'FIXED_NONLINEAR_MODEL')
            invariant,coverage=direction_supported(atlas)
            nonlinear_supported,nonlinear_coverage=direction_supported(nonlinear_atlas)
            scores={k:ranking(vy,p,cache.meta['time_online'][va],vids) for k,p in preds.items()}
            gain=cluster_difference(vids,binary_losses(vy,preds['CONTEXT_ONLY']),binary_losses(vy,preds['O1']))
            nonlinear=bool(nonlinear_supported and scores['O1']['TS_AUC']-scores['O0']['TS_AUC']>.002 and gain['ci_low'] is not None and gain['ci_low']>0 and gain['gain']>.0001)
            conditional=conditional_models(cache,j,scope,tr,va,directory/f'F{j:03d}_conditional') if j in DETAILED else {'status':'NOT_IN_FIXED_DETAILED_MENU'}
            record={'feature':j,'feature_id':f'NX8_F{j:03d}','name':name,'scope':scope.provenance(),
                'direction':direction,'invariant_direction_supported':invariant,'fixed_nonlinear_supported':nonlinear,
                'domain_coverage':coverage,'nonlinear_domain_coverage':nonlinear_coverage,
                'scores':scores,'O1_increment_beyond_context':gain,'atlas':atlas+nonlinear_atlas,
                'conditional':conditional,'all_converged':all(m.training.get('converged',True) for m in models.values()),
                'timestamp':utc_now(),'selection_labels_from_scope_only':True}
            atomic(path,record);records.append(record)
            if j%16==0:print(utc_now(),'SCREEN PROGRESS',scope.name,j+1,len(NAMES),flush=True)
    # All conditional contrasts in this split form one finite hypothesis family.
    tested=[(r['feature'],k,c['p_value']) for r in records if r['conditional']['status']=='COMPLETE'
            for k,c in enumerate(r['conditional']['contrasts'])]
    adjusted=holm([p for _,_,p in tested])
    corrections=[{'feature':j,'contrast_index':k,'raw_p':p,'Holm_p':float(a)} for (j,k,p),a in zip(tested,adjusted)]
    atomic(directory/'MULTIPLICITY.json',{'family':'all eligible12 x3 direct contrasts in this training split',
        'tests':corrections,'training_only':True,'failed_or_unsupported_features_also_retained':True})
    atomic(directory/'DONE.json',{'status':'COMPLETE','timestamp':utc_now(),'scope':scope.provenance(),
        'features':len(records),'conditional_eligible':len(DETAILED),'multiplicity_sha256':digest(directory/'MULTIPLICITY.json'),
        'records_sha256':{f'F{r["feature"]:03d}.json':digest(directory/f'F{r["feature"]:03d}.json') for r in records},
        'scores_scope':'inner or subinner heldout entirely inside parent training; no outer score'})
    return records,corrections


def summarize_screen(panels,corrections,parent):
    classifications=[]
    for j,name in enumerate(NAMES):
        rows=[p[j] for p in panels]
        inv=all(r['invariant_direction_supported'] and r['all_converged'] for r in rows) and len({r['direction'] for r in rows})==1
        nonlinear=all(r['fixed_nonlinear_supported'] and r['all_converged'] for r in rows)
        conditional_support=[]
        for r,corr in zip(rows,corrections):
            c=r['conditional']
            if c['status']!='COMPLETE':conditional_support.append(False);continue
            significance=any(v['feature']==j and v['Holm_p']<.05 for v in corr)
            benefit=c['comparisons']['O1_STATE_MAIN'];state_benefit=c['comparisons']['STATE_ONLY']
            domains=c['heldout_domains']
            temporal=sum(d['domain_type']=='time' and d['gain']>0 for d in domains)
            background=sum(d['domain_type']=='H_length' and d['gain']>0 for d in domains)
            conditional_support.append(bool(significance and c['converged'] and c['opposing_train_fixed_derivatives_on_common_range'] and
                benefit['ci_low'] is not None and benefit['ci_low']>0 and benefit['gain']>.0001 and
                state_benefit['ci_low'] is not None and state_benefit['ci_low']>0 and temporal>=2 and background>=2))
        # A different reversing pair in each fold is not a reproducible rule.
        # Require the same state pair, direction and feature-value anchor in
        # an overlapping training-fitted domain, plus repeated time/H domains.
        from itertools import combinations
        reproducible=[]
        supported_folds=[k for k,yes in enumerate(conditional_support) if yes]
        for count in (3,2):
            for members in combinations(supported_folds,count):
                cond=[rows[k]['conditional'] for k in members]
                low=max(c['domain'][0] for c in cond);high=min(c['domain'][1] for c in cond)
                if high-low<=1e-3:continue
                anchor=float(np.clip(0.,low,high))
                domains=[{(d['domain_type'],d['domain']) for d in c['heldout_domains'] if d['gain']>0} for c in cond]
                common_domains=set.intersection(*domains)
                if any(sum(k==kind for k,_ in common_domains)<2 for kind in ('time','H_length')):continue
                for contrast_index,(a,b) in enumerate(((0,1),(2,1),(0,2))):
                    signs=[]
                    for fold_index,c in zip(members,cond):
                        adjusted=[v['Holm_p'] for v in corrections[fold_index] if v['feature']==j and v['contrast_index']==contrast_index]
                        if not adjusted or adjusted[0]>=.05:break
                        w=c['derivative_weights_raw_design'];z=anchor/4.
                        fixed=w.get('F_linear',0.)+2*z*w.get('F_square',0.)
                        fixed+=w.get('F_positive',0.) if z>0 else -w.get('F_negative',0.) if z<0 else 0.
                        sa=np.sign(fixed+w.get(f'own_state_{a}_times_F',0.))
                        sb=np.sign(fixed+w.get(f'own_state_{b}_times_F',0.))
                        if sa*sb>=0:break
                        signs.append((int(sa),int(sb)))
                    if len(signs)==count and len(set(signs))==1:
                        reproducible.append({'inner_folds':list(members),'state_pair':[a,b],
                            'directions':list(signs[0]),'F_anchor':anchor,'common_F_domain':[low,high],
                            'common_time_H_domains':[list(v) for v in sorted(common_domains)]})
        conditional_count=max((len(p['inner_folds']) for p in reproducible),default=0)
        if conditional_count>=2:label='CONDITIONAL_VARIABLE_CANDIDATE'
        elif nonlinear:label='STABLE_NONLINEAR'
        elif inv:label='INVARIANT_DIRECTION'
        else:label='UNRESOLVED'
        # A null test's lack of significance is deliberately not UNINFORMATIVE.
        evidence={'feature':j,'feature_id':f'NX8_F{j:03d}','name':name,'classification':label,
            'inner_folds':3,'conditional_supported_folds':conditional_count,
            'conditional_gain_supported_folds_before_rule_consistency':int(sum(conditional_support)),
            'repeated_conditional_rules':reproducible,
            'signed_direction':[r['direction'] for r in rows],
            'mean_fixed_nonlinear_AUC':float(np.mean([r['scores']['O1']['TS_AUC'] for r in rows])),
            'mean_conditional_gain':float(np.mean([r['conditional']['comparisons']['O1_STATE_MAIN']['gain'] for r in rows if r['conditional']['status']=='COMPLETE'])) if any(r['conditional']['status']=='COMPLETE' for r in rows) else None,
            'certified':False,'scope':parent.provenance(),'domain_rule':'Practical direction means excluding >.03 AUC reversal in supported domains; no claim of equal effect magnitude or arbitrary-domain invariance'}
        classifications.append(evidence)
    stable=sorted([r for r in classifications if r['classification'] in ('INVARIANT_DIRECTION','STABLE_NONLINEAR')],key=lambda r:(-r['mean_fixed_nonlinear_AUC'],r['feature_id']))[:6]
    variables=sorted([r for r in classifications if r['classification']=='CONDITIONAL_VARIABLE_CANDIDATE'],key=lambda r:(-r['mean_conditional_gain'],r['feature_id']))[:6]
    return {'classifications':classifications,'stable_features':[r['feature'] for r in stable],
            'variable_features':[r['feature'] for r in variables],
            'detailed_features':list(DETAILED),'scope':parent.provenance(),
            'selection_ids':sorted(parent.training_ids),'outer_or_meta_validation_labels_used':False,
            'no_support_fallback':'Empty feature sets stay empty; no forced candidates'}
