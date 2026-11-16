"""Analytic single-change prior pressure; no learned-model or raw-data score."""
from pathlib import Path
import sys
import math
import numpy as np
import pandas as pd
from scipy.special import logsumexp
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,atomic_json,digest,utc_now,progress
from src.next4 import sequential as d


def main():
    out=BASE/'hazard_pressure';out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'START.json',{'created_utc':utc_now(),'source_sha256':digest(Path(__file__)),
        'sequential_source_sha256':digest(ROOT/'src/next4/sequential.py'),'new_model_fits':0},immutable=True)
    rows=[]
    alternatives=[('mean',float(mu),.5*float(mu)**2) for mu in d.MEANS]
    alternatives += [('variance',float(v),.5*(math.log(float(v))-1+1/float(v))) for v in d.VARIANCES]
    for hazard in d.HAZARDS:
        pressure=-math.log1p(-float(hazard))
        for kind,parameter,kl in alternatives:
            rows.append({'hazard':float(hazard),'alternative_kind':kind,'alternative_parameter':parameter,
                'ideal_gaussian_null_KL_to_alternative':kl,'no_change_prior_log_odds_pressure_per_point':pressure,
                'expected_log_multiplier_of_existing_alternative_odds':pressure-kl,
                'positive_expected_log_multiplier':pressure>kl,
                'critical_hazard_for_this_alternative':-math.expm1(-kl)})
    pd.DataFrame(rows).to_csv(out/'IDEAL_NULL_LOG_MULTIPLIERS.csv',index=False)
    # Independent enumeration of every possible single change time, not another
    # recursive filter: odds_k(t)=sum_s h/K*(1-h)^(s-1-t)*prod_{u=s}^t LR_k(u).
    states=np.full((3,11),-np.inf);states[:,0]=0.;output=np.empty(20);zs=[];error=0.;enumeration=[]
    for t in range(1,129):
        z=1. if t%2 else -1.;zs.append(z);d.hazard_features(z,states,output)
        if t not in (1,4,32,128):continue
        a=np.asarray(zs);ll=np.column_stack([a[:,None]*d.MEANS-.5*d.MEANS**2,
            -.5*np.log(d.VARIANCES)+.5*(1-1/d.VARIANCES)*a[:,None]**2])
        suffix=np.cumsum(ll[::-1],axis=0)[::-1];start=np.arange(t)
        for j,hazard in enumerate(d.HAZARDS):
            terms=math.log(float(hazard)/10)+(start-t)[:,None]*math.log1p(-float(hazard))+suffix
            expected=logsumexp(terms,axis=0);actual=states[j,1:]-states[j,0]
            err=float(np.max(np.abs(actual-expected)));error=max(error,err)
            np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-11)
            enumeration.append({'time':t,'hazard':float(hazard),'maximum_log_odds_error':err})
    pd.DataFrame(enumeration).to_csv(out/'CHANGE_TIME_ENUMERATION_ORACLE.csv',index=False)
    # Continue the same deterministic zero-mean unit-second-moment feature toy.
    # It is not a fitted competition null or a random Gaussian cohort.
    toy=[]
    for t in range(129,16385):
        d.hazard_features(1. if t%2 else -1.,states,output)
        if t not in (512,1024,4096,16384):continue
        for j,hazard in enumerate(d.HAZARDS):
            logodds=float(logsumexp(states[j,1:])-states[j,0])
            toy.append({'time':t,'hazard':float(hazard),'any_change_log_odds':logodds,
                'any_change_posterior_working_toy':float(-math.expm1(float(states[j,0]))),
                'exposed_clipped_log_odds_feature':float(output[6*j]),
                'closest_positive_mean_log_odds':float(states[j,4]-states[j,0]),
                'closest_negative_mean_log_odds':float(states[j,3]-states[j,0])})
    pd.DataFrame(toy).to_csv(out/'ALTERNATING_UNIT_MOMENT_TOY.csv',index=False)
    neutral=[]
    for t in (32,128,512,1024,4096,16384):
        for hazard in d.HAZARDS:
            exponent=-t*math.log1p(-float(hazard))
            neutral.append({'time':t,'hazard':float(hazard),'all_LR_equal_one_changed_probability':-math.expm1(-exponent),
                'all_LR_equal_one_log_odds':exponent+math.log1p(-math.exp(-exponent))})
    pd.DataFrame(neutral).to_csv(out/'NEUTRAL_LIKELIHOOD_BOUNDARY.csv',index=False)
    atomic_json(out/'RESULTS.json',{'status':'PASS','created_utc':utc_now(),'new_model_fits':0,'new_model_scores':0,
        'raw_or_DEV_or_seal_rows':0,'new_random_seeds':0,'deterministic_feature_toys':1,
        'independent_change_time_enumeration_maximum_error':error,
        'positive_ideal_null_log_multiplier_count':sum(r['positive_expected_log_multiplier'] for r in rows),
        'interpretation':'For each absorbing alternative, R_tk = LR_tk*(R_(t-1)k+h/K)/(1-h). Its existing-odds log multiplier under ideal iid N(0,1) is -KL(null||alternative)-log(1-h). With h=1/32 and mean +/-0.25 this is slightly positive. This is an analytic property of the frozen working model, not a measured FPR or a claim about fitted whitened/clipped real residuals. The alternating feature toy is deterministic; the neutral-LR calculation is an abstract boundary, not an additional candidate. Real and calibrated synthetic evidence remain authoritative.',
        'BOCPD_scope_source':'https://arxiv.org/abs/0710.3742',
        'source_scope':'Adams-MacKay models run length and repeated changes; NEXT4 D3 is a finite absorbing single-change working model. The odds-pressure derivation here is from local source algebra, not a result attributed to that paper.',
        'model_or_hazard_changes':False,'gate_changes':False,
        'files_sha256':{p.name:digest(p) for p in out.glob('*.csv')}},immutable=True)
    progress('D3 hazard-prior pressure analyzed','Independent change-time enumeration passed; ideal Gaussian drift and a deterministic unit-moment feature toy separated from fitted-model null FPR')
    print('NEXT4 HAZARD PRIOR PRESSURE COMPLETE',flush=True)


if __name__=='__main__':main()
