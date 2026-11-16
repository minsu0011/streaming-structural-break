"""Two bounded H-background orientation conditions, fitted within training IDs.

H length alone does not audit H dynamics/tails. This explicit protocol
supplement preserves the initial selector and its chronology. No outer score
has been used to choose these two observable background coordinates.
"""
from dataclasses import dataclass
from pathlib import Path
import copy
import numpy as np
import joblib
from src.next8.common import BASE,atomic,read_json,digest,utc_now,activity,ids_hash
from src.next8.features import NAMES,DETAILED,UNKNOWN
from src.next8.learning import select_training_only,orientation_design,fit_logistic
from src.next8.screening import (conditional_models,sampled_rows,context_for,
    _oriented_rank,summarize_screen)
from src.next8.statistics import holm,ranking
from src.next.fitting import weights_from_training

CONTEXT_MENU={'H_LAG1':2,'H_TAIL3':4}
STATE_ORDER=('OWN_F','H_LAG1','H_TAIL3')


@dataclass(frozen=True)
class HStateSpec:
    kind: str
    context_index: int
    low: float
    high: float
    minimum: float
    maximum: float
    valid: bool
    training: dict

    @classmethod
    def fit(cls,cache,scope,kind):
        if kind not in CONTEXT_MENU:raise ValueError('Only two fixed H-background conditions')
        ids=cache.series['dataset_id']
        take=np.isin(ids,list(scope.training_ids))
        selected=ids[take]
        scope.admit_training(selected)
        if set(map(int,selected))!=scope.training_ids:raise ValueError('Complete training H membership required')
        select_training_only({'kind':kind},scope,selected,'state')
        index=CONTEXT_MENU[kind];x=np.asarray(cache.series['context'][take,index],dtype=float)
        if not np.isfinite(x).all():raise ValueError('Finite H context required')
        low,high=np.quantile(x,[1/3,2/3])
        valid=bool(len(x)>=64 and high-low>1e-8)
        return cls(kind,index,float(low),float(high),float(x.min()),float(x.max()),valid,
                   {**scope.provenance(),'H_series_count':len(x),'H_quantiles_training_only':True,
                    'equal_weight_per_H_series':True,'Online_values_or_labels_used':False})

    def states_from_context(self,context,feature_valid=True):
        x=np.asarray(context)[...,self.context_index]
        known=np.isfinite(x)&(x>=self.minimum)&(x<=self.maximum)&self.valid&np.asarray(feature_valid)
        return np.where(known,np.where(x<self.low,0,np.where(x>self.high,2,1)),UNKNOWN).astype(np.int8)


class HStateCache:
    """Proxy changes only the observable condition, never F or its history."""
    def __init__(self,base,spec):self.base=base;self.spec=spec
    def __getattr__(self,name):return getattr(self.base,name)
    def states(self,rows,columns,*,previous=False):
        rows=np.asarray(rows,dtype=np.int64);columns=np.asarray(columns,dtype=np.int32)
        s=self.base.series_index[rows]
        context=self.base.series['context'][s]
        common=self.spec.states_from_context(context)
        valid=self.base.series['valid'][s[:,None],columns[None,:]]
        return np.where(valid,common[:,None],UNKNOWN).astype(np.int8)


def h_atlas(proxy,rows,j,prediction,direction,mode):
    states=proxy.states(rows,[j],previous=True)[:,0]
    y=proxy.meta['target'][rows];t=proxy.meta['time_online'][rows];ids=proxy.meta['dataset_id'][rows]
    output=[]
    for state in range(4):
        keep=states==state
        if not np.any(keep):continue
        output.append({'feature':j,'feature_id':f'NX8_F{j:03d}','state_estimator':proxy.spec.kind,
            'state':state,'mode':mode,'condition_is_H_only_constant_within_series':True,
            'evaluation':'INNER_HELD_OUT_INSIDE_PARENT_TRAIN',
            **_oriented_rank(y[keep],prediction[keep],t[keep],ids[keep],direction)})
    return output


def audit_h_split(cache,scope,old_directory,directory,stable_shortlist,cutoff=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    tr=sampled_rows(cache,scope.training_ids,cutoff);va=sampled_rows(cache,scope.validation_ids,cutoff)
    specs={k:HStateSpec.fit(cache,scope,k) for k in CONTEXT_MENU}
    proxies={k:HStateCache(cache,s) for k,s in specs.items()}
    spec_values={k:vars(s) for k,s in specs.items()}
    if (directory/'H_STATE_ESTIMATORS.json').exists():
        assert read_json(directory/'H_STATE_ESTIMATORS.json')==spec_values
    else:atomic(directory/'H_STATE_ESTIMATORS.json',spec_values)
    rows=[];branches={k:[] for k in STATE_ORDER}
    ty=cache.meta['target'][tr];vy=cache.meta['target'][va]
    tids=cache.meta['dataset_id'][tr];vids=cache.meta['dataset_id'][va]
    weights=weights_from_training(ty,cache.meta['time_online'][tr],tids,'W3')
    for j in range(len(NAMES)):
        old=read_json(Path(old_directory)/f'F{j:03d}.json')
        assert old['scope']==scope.provenance()
        branches['OWN_F'].append(old)
        p=directory/f'F{j:03d}.json'
        if p.exists():
            r=read_json(p);rows.append(r)
            for kind in CONTEXT_MENU:
                b=copy.deepcopy(old);b['conditional']=r['conditional'][kind];branches[kind].append(b)
            continue
        with activity(f'H-background rule audit {scope.name} F{j:03d}','H_CONTEXT_ORIENTATION','B5_DOMAIN_GENERALIZATION'):
            vf=cache.current(va,[j])[:,0]
            atlas=[]
            for kind,proxy in proxies.items():atlas.extend(h_atlas(proxy,va,j,vf,old['direction'],'SIGNED_F'))
            # Reconstruct only a retained stable O1 model where the initial
            # stage did not save coefficients. This serves NEW H-background
            # validation, with exact agreement against its saved old score.
            nonlinear=[];reconstructed=None
            if j in stable_shortlist:
                train=orientation_design(cache.current(tr,[j])[:,0],cache.states(tr,[j],previous=True)[:,0],context_for(cache,tr,j),tids,'O1')
                validation=orientation_design(vf,cache.states(va,[j],previous=True)[:,0],context_for(cache,va,j),vids,'O1')
                model=fit_logistic(train,ty,scope,weights)
                predictions=model.predict_proba(validation)[:,1]
                score=ranking(vy,predictions,cache.meta['time_online'][va],vids)['TS_AUC']
                if abs(score-old['scores']['O1']['TS_AUC'])>1e-12:raise RuntimeError('Reconstructed initial O1 score differs')
                path=directory/f'F{j:03d}_O1_reconstructed.joblib'
                if path.exists():raise FileExistsError(path)
                joblib.dump(model,path);reconstructed={'path':path.relative_to(BASE).as_posix(),'sha256':digest(path),'old_score_parity':'PASS','purpose':'New H-background audit, not repetition of old performance claim'}
                for kind,proxy in proxies.items():nonlinear.extend(h_atlas(proxy,va,j,predictions,1,'FIXED_NONLINEAR'))
            conditional={}
            for kind,proxy in proxies.items():
                c=conditional_models(proxy,j,scope,tr,va,directory/f'F{j:03d}_{kind}') if j in DETAILED else {'status':'NOT_IN_FIXED_DETAILED_MENU'}
                c['orientation_condition']=kind
                c['condition_fitted_from_training_H_only']=True
                c['not_a_within_series_orientation_switch']=True
                conditional[kind]=c
                b=copy.deepcopy(old);b['conditional']=c;branches[kind].append(b)
            r={'feature':j,'scope':scope.provenance(),'H_atlas':atlas,'nonlinear_H_atlas':nonlinear,
                'conditional':conditional,'reconstructed_O1':reconstructed,'timestamp':utc_now()}
            atomic(p,r);rows.append(r)
    tested=[(kind,r['feature'],k,c['p_value']) for kind,panel in branches.items() for r in panel
            if r['conditional']['status']=='COMPLETE' for k,c in enumerate(r['conditional']['contrasts'])]
    adjusted=holm([r[-1] for r in tested])
    corrections={kind:[] for kind in STATE_ORDER}
    for (kind,j,k,p),a in zip(tested,adjusted):
        corrections[kind].append({'feature':j,'contrast_index':k,'raw_p':p,'Holm_p':float(a)})
    atomic(directory/'MULTIPLICITY.json',{'family':'OWN_F + H_LAG1 + H_TAIL3, all12 eligible features x3 state contrasts in this training split','branches':corrections})
    atomic(directory/'DONE.json',{'status':'COMPLETE','timestamp':utc_now(),'scope':scope.provenance(),
        'records_sha256':{f'F{r["feature"]:03d}.json':digest(directory/f'F{r["feature"]:03d}.json') for r in rows},
        'H_state_estimators_sha256':digest(directory/'H_STATE_ESTIMATORS.json'),'before_outer_evaluation':True})
    return rows,branches,corrections


def h_direction_ok(rows,nonlinear=False):
    atlas=rows['nonlinear_H_atlas'] if nonlinear else rows['H_atlas']
    supported=[r for r in atlas if r['state']!=UNKNOWN and r['support_status']=='SUPPORTED']
    coverage={kind:sum(r['state_estimator']==kind for r in supported) for kind in CONTEXT_MENU}
    return bool(max(coverage.values(),default=0)>=2 and all(r['ci_low']>.47 for r in supported)),coverage


def combine_selection(cache,scope,old_selection,all_h_rows,all_branches,all_corrections):
    summaries={kind:summarize_screen([p[kind] for p in all_branches],[p[kind] for p in all_corrections],scope) for kind in STATE_ORDER}
    final=[];state_specs={}
    for j,name in enumerate(NAMES):
        old=old_selection['classifications'][j]
        candidates=[(kind,summaries[kind]['classifications'][j]) for kind in STATE_ORDER if summaries[kind]['classifications'][j]['classification']=='CONDITIONAL_VARIABLE_CANDIDATE']
        if candidates:
            kind,best=sorted(candidates,key=lambda p:(-p[1]['mean_conditional_gain'],STATE_ORDER.index(p[0])))[0]
            record={**best,'selected_orientation_state':kind,'classification':'CONDITIONAL_VARIABLE_CANDIDATE',
                    'candidate_state_families_tested':list(STATE_ORDER)}
        else:
            keep_old=old['classification'] in ('INVARIANT_DIRECTION','STABLE_NONLINEAR')
            checked=[]
            for rows in all_h_rows:
                passed,coverage=h_direction_ok(rows[j],nonlinear=old['classification']=='STABLE_NONLINEAR')
                checked.append({'passed':passed,'coverage':coverage})
            keep_old &= all(x['passed'] for x in checked)
            record={**old,'classification':old['classification'] if keep_old else 'UNRESOLVED',
                'selected_orientation_state':'OWN_F','H_background_checks':checked,
                'H_nonlinear_retest_scope':'At most6 retained stable shortlist; unretained nonlinear features are not certified by unrun H tests'}
        record['protocol_supplement']='H_LAG1_AND_H_TAIL3_TRAIN_FITTED'
        final.append(record)
    # Preserve the finite original six-feature stable shortlist; failures are
    # not filled by repeatedly walking down an unlimited ranked list.
    stable=[j for j in old_selection['stable_features'] if final[j]['classification'] in ('INVARIANT_DIRECTION','STABLE_NONLINEAR')]
    variables=sorted([r for r in final if r['classification']=='CONDITIONAL_VARIABLE_CANDIDATE'],
                     key=lambda r:(-r['mean_conditional_gain'],r['feature_id']))[:6]
    for r in variables:
        kind=r['selected_orientation_state']
        state_specs[str(r['feature'])]=({'kind':'OWN_F','policy':'H-only pre-current feature q20/q80; no population fit'} if kind=='OWN_F' else vars(HStateSpec.fit(cache,scope,kind)))
    return {'status':'FROZEN','timestamp':utc_now(),'classifications':final,'stable_features':stable,
        'variable_features':[r['feature'] for r in variables],'orientation_state_specs':state_specs,
        'scope':scope.provenance(),'selection_ids':sorted(scope.training_ids),'detailed_features':list(DETAILED),
        'parent_full_initial_selection_sha256':None,'outer_or_meta_validation_labels_used':False,
        'H_state_can_be_constant_within_series':True,'H_conditional_direction_not_claimed_as_within_series_switch':True,
        'stable_shortlist_not_refilled_after_domain_failures':True}
