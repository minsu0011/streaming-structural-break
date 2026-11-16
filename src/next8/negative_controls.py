"""Fixed interpretable synthetic controls; no RP outcome or model tuning."""
import numpy as np
from sklearn.metrics import roc_auc_score
from src.scoring.ts_auc import ts_auc
from src.next8.learning import Design,orientation_design,fit_logistic
from src.next8.data import FitScope
from src.next8.statistics import rank_influence,regression_metrics


def fixed_controls():
    seed=8101604;rng=np.random.default_rng(seed);records=[];arrays={}
    n=256;length=128;ids=np.repeat(np.arange(n),length);t=np.tile(np.arange(length),n)
    tau=16+(np.arange(n)*17%96);y=(t>=tau[ids]).astype(np.uint8)
    time_score=t.astype(float)
    pooled=float(roc_auc_score(y,time_score));matched=float(ts_auc(y,time_score,t))
    assert matched==.5 and pooled>.7
    records.append({'control':'TIME_POOLED_ASSOCIATION','status':'EXPECTED_FAILURE_EXPOSED','pooled_row_AUC':pooled,
        'same_time_TS_AUC':matched,'mechanism':'Absolute elapsed time ranks different stages when pooled; every same-time score ties',
        'competition_labels_monotone_within_series':True,'time_used_as_legitimate_observable_input':True,
        'claim_limit':'Shows scorer confounding, not a measured defect in any RP model'})
    # A fixed U shape with balanced signs and valid monotone break labels.
    sign=np.where(np.arange(n)%2,1.,-1.);high=(np.arange(n)//2)%2==1
    f=sign[ids]*np.where(high[ids],2.,.5);uy=(t>=np.where(high[ids],16,112)).astype(np.uint8)
    own=np.where(f<0,0,2).astype(np.int8);scope=FitScope(frozenset(range(128)),frozenset(range(128,256)),frozenset(),'fixed_U_control')
    train=ids<128;valid=~train;pred={};training={}
    for method in ('O0','O1','O2'):
        td=orientation_design(f[train],own[train],np.zeros((train.sum(),1),dtype=np.float32),ids[train],method)
        vd=orientation_design(f[valid],own[valid],np.zeros((valid.sum(),1),dtype=np.float32),ids[valid],method)
        model=fit_logistic(td,uy[train],scope);pred[method]=model.predict_proba(vd)[:,1];training[method]=model.training
    scores={m:float(ts_auc(uy[valid],p,t[valid])) for m,p in pred.items()}
    assert scores['O1']>.99 and scores['O2']-scores['O1']<1e-12
    records.append({'control':'FIXED_U_SHAPE_NOT_DYNAMIC','status':'FIXED_NONLINEAR_EXPLAINS_RELATION','TS_AUC':scores,
        'raw_F_TS_AUC':float(ts_auc(uy[valid],f[valid],t[valid])),'own_state_is_constant_within_series':True,
        'generating_law':'Balanced positive/negative magnitudes; high absolute magnitude has earlier fixed break onset',
        'fitted_control_models':3,'training':training,'no_dynamic_orientation_increment_over_fixed_nonlinearity':True})
    arrays.update({'U_feature':f[valid],'U_time':t[valid],'U_target':uy[valid],**{'U_'+m:p for m,p in pred.items()}})
    # Varying survivors change the population mean without a single F moving.
    h=np.arange(n)%2==0;lengths=np.where(h,32,128);sids=np.repeat(np.arange(n),lengths)
    st=np.concatenate([np.arange(k) for k in lengths]);sf=np.where(h[sids],2.,-1.)
    means=np.array([sf[st==k].mean() for k in range(128)])
    assert means[0]==.5 and means[32]==-1.
    records.append({'control':'SURVIVOR_COMPOSITION','status':'COMPOSITION_SHIFT_NOT_FEATURE_MOVEMENT',
        'population_mean_t0':float(means[0]),'population_mean_t32':float(means[32]),
        'maximum_within_series_feature_change':0.,'short_series':128,'long_series':128,
        'total_online_length_not_a_predictive_input':True,'mechanism':'High constant-F series stop early; remaining constant-F series have lower values'})
    arrays['SURVIVOR_MEAN']=means
    # Same-raw EWMA recurrence is exact; large self predictability is designed.
    raw=rng.normal(size=(256,256));a=np.zeros_like(raw);b=np.zeros_like(raw)
    for k in range(256):
        a[:,k]=.9*(a[:,k-1] if k else 0)+.1*raw[:,k]
        b[:,k]=.95*(b[:,k-1] if k else 0)+.05*raw[:,k]
    residual_a=a[:,1:]-.9*a[:,:-1];residual_b=b[:,1:]-.95*b[:,:-1]
    exact=float(np.max(np.abs(residual_a-2*residual_b)));assert exact<1e-12
    future=a[:,1:].ravel();own=.9*a[:,:-1].ravel();zero=np.zeros(len(future));g=np.repeat(np.arange(256),255)
    correlation=float(np.corrcoef(a.ravel(),b.ravel())[0,1])
    ynull=(np.tile(np.arange(1,256),256)>=rng.integers(32,224,256)[g]).astype(np.uint8)
    rank,_,_=rank_influence(ynull,future,np.tile(np.arange(1,256),256),g)
    records.append({'control':'SELF_RECURRENCE_AND_SHARED_RAW','status':'MECHANICAL_RELATION_CONFIRMED',
        'own_prediction_MSE':regression_metrics(future,own,g)['MSE'],'zero_mean_MSE':regression_metrics(future,zero,g)['MSE'],
        'feature_correlation':correlation,'shared_innovation_identity_max_error':exact,
        'independent_random_onset_TS_AUC':rank,'generating_equations':['A[t]=.9A[t-1]+.1x[t]','B[t]=.95B[t-1]+.05x[t]'],
        'no_causal_factor_claim':True,'no_break_specificity_from_self_prediction':True})
    arrays.update({'EWMA_A':a[0],'EWMA_B':b[0],'EWMA_SHARED_RAW':raw[0]})
    for name in ('future_target_state','validation_label_sign','true_phase','tau','total_online_length'):
        rejected=False
        try:Design(np.zeros((2,1),dtype=np.float32),(name,),np.arange(2)).validate()
        except ValueError:rejected=True
        assert rejected
        records.append({'control':name.upper(),'status':'REJECTED_UNOBSERVABLE_SCHEMA',
            'forbidden_values_constructed_or_given_to_model':False,'schema_test_alone_proves_causality':False,
            'additional_evidence_required':'Actual prefix/future-suffix perturbation and label-access-bomb tests of each production path'})
    return {'status':'COMPLETE','seed':seed,'records':records,'source_data':'Fixed synthetic controls only',
        'real_data_or_protected_asset_access':False,'control_recipe_predates_outer_results':True,
        'three_toy_fits_are_controls_not_new_M_candidates':True},arrays
