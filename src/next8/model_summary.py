"""Cross-fold paired reporting under the unchanged historical promotion gate."""
import numpy as np
from src.next8.common import BASE,read_json,digest
from src.next8.paired_bootstrap import summarize_paired
from src.next8.evaluation import METHODS,simultaneous_envelope

BASELINE_FOLDS=(.6289166309810504,.613008744689797,.6208808016668448,.6176194712143422,.6214004017972592)


def load_domain(fold,name,early=False):
    scope=f'outer_{fold}'+('_EARLY_T199' if early else '')
    p=BASE/'model_evaluation'/scope/name;r=read_json(p/'DONE.json')
    assert digest(p/'BOOTSTRAP.npz')==r['bootstrap_sha256']
    with np.load(p/'BOOTSTRAP.npz',allow_pickle=False) as a:samples={k:a[k].copy() for k in a.files}
    return r,samples


def numerical_promotion(comparison,major_collapse):
    if comparison.get('status')!='SUPPORTED':return False
    return bool(comparison['mean_delta']>=.0012 and comparison['ci_low']>0 and comparison['positive_folds']>=4 and
        comparison['worst_fold_delta']>=-.001 and not major_collapse)


def summarize_models(early=False):
    primary=[load_domain(f,'ALL',early) for f in range(5)]
    points=[r['points'] for r,_ in primary];samples=[a for _,a in primary]
    if not early:
        actual=np.asarray([p['M0']['TS_AUC'] for p in points]);assert np.max(np.abs(actual-BASELINE_FOLDS))<1e-12
        assert abs(actual.mean()-.6203652100698587)<1e-12
    compared=[];domain_rows=[];fold_rows=[]
    names=read_json(BASE/'model_evaluation'/('outer_0_EARLY_T199' if early else 'outer_0')/'DONE.json')['domains']
    collapsed={m:[] for m in METHODS if m!='M0'}
    for name in names:
        if name=='ALL':continue
        loaded=[(f,*load_domain(f,name,early)) for f in range(5)]
        admitted=[(f,r,a) for f,r,a in loaded if r['resampling']['status']=='SUPPORTED' and r['resampling']['support']['major_domain_supported']]
        for candidate in METHODS[1:]:
            result=summarize_paired([r['points'] for _,r,_ in admitted],[a for _,_,a in admitted],'M0',candidate) if admitted else {'status':'INSUFFICIENT_SUPPORT'}
            collapse=result.get('status')=='SUPPORTED' and result['mean_delta']<-.01 and result['ci_high']<0
            if collapse:collapsed[candidate].append(name)
            domain_rows.append({'early_only':early,'domain':name,'domain_family':loaded[0][1]['family'],'candidate':candidate,
                'supported_fold_count':len(admitted),'supported_folds':'|'.join(str(f) for f,_,_ in admitted),
                'major_collapse':collapse,**result,
                'support_by_fold':[{'fold':f,**r['resampling']['support']} for f,r,_ in loaded]})
    for f,(r,_) in enumerate(primary):
        for method in METHODS:
            fold_rows.append({'early_only':early,'fold':f,'method':method,'TS_AUC':r['points'][method]['TS_AUC'],
                'delta_BL80':(r['points'][method]['TS_AUC']-r['points']['M0']['TS_AUC']) if r['points'][method]['TS_AUC'] is not None and r['points']['M0']['TS_AUC'] is not None else None,
                'same_time_pairs':r['points'][method]['pairs'],'M6_eligible':r['prediction_provenance']['integration_eligible'],
                'M6_status':r['prediction_provenance']['integration_status'],'source_prediction_sha256':r['prediction_provenance']['prediction_sha256']})
    contrasts=[('M0',m) for m in METHODS[1:]]+[('M2','M5'),('M3','M5'),('M4','M5'),
        ('BL95_M3_05_CONTROL','M6'),('BL95_M4_05_CONTROL','M6'),('BL95_M2_05_CONTROL','M6')]
    for reference,candidate in contrasts:
        result=summarize_paired(points,samples,reference,candidate)
        compared.append({'early_only':early,'reference':reference,'candidate':candidate,**result,'major_collapse_domains':collapsed.get(candidate,[]),
            'historical_numeric_promotion_gate':numerical_promotion(result,bool(collapsed.get(candidate))) if reference=='M0' and candidate.startswith('M') else None,
            'promotion_final':False,'promotion_status':'AWAITING_CAUSAL_RUNTIME_REPRODUCTION_AND_FINAL_AUDIT',
            'new_primary_role_assigned':False})
    means={m:float(np.mean([p[m]['TS_AUC'] for p in points])) if all(p[m]['TS_AUC'] is not None for p in points) else None for m in METHODS}
    return {'status':'COMPLETE','early_only':early,'mean_TS_AUC':means,'comparisons':compared,'folds':fold_rows,
        'domains':domain_rows,'simultaneous_family_envelope':simultaneous_envelope(points,samples),
        'baseline':'EARLY_T199_REFIT_BL80' if early else 'ORIGINAL_EXCLUDED_GROUP_RP_BL80',
        'five_procedures_equal_weight':True,'all_models_kept_including_aliases':True,
        'graph_superiority_not_a_retroactive_new_historical_promotion_gate':True,
        'same_input_or_same_BL_tie_limits_graph_specific_claim':True,
        'early_M6_normal_and_early_eligibility_are_distinct':True}
