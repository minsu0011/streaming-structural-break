"""Known-truth density control: a fixed U-shape can mimic raw sign reversal."""
import numpy as np
from scipy.special import expit


def expected_auc(probability,score,weight):
    order=np.argsort(score,kind='stable');p=np.asarray(probability)[order];s=np.asarray(score)[order];w=np.asarray(weight)[order]
    positive=w*p;negative=w*(1-p);cuts=np.r_[0,np.flatnonzero(np.diff(s))+1,len(s)]
    wins=0.;previous=0.
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        a,b=positive[lo:hi].sum(),negative[lo:hi].sum();wins+=a*(previous+.5*b);previous+=b
    return float(wins/(positive.sum()*negative.sum()))


def fixed_u_shape():
    positive=np.linspace(.002,2.,1000);f=np.r_[-positive[::-1],0.,positive]
    density={s:np.exp(s*1.2*f) for s in [-1,1]}
    ranges={}
    for s,w in density.items():
        cdf=np.cumsum(w)/w.sum();ranges[s]=np.interp([.1,.9],cdf,f)
    low=max(r[0] for r in ranges.values());high=min(r[1] for r in ranges.values())
    keep=(f>=low)&(f<=high);f=f[keep];g=expit(2.*(f*f-1.))
    result=[]
    for s in [-1,1]:
        w=np.exp(s*1.2*f);w/=w.sum();prior=float(np.sum(w*g))
        loss=float(np.sum(w*(-g*np.log(g)-(1-g)*np.log1p(-g))))
        prior_loss=float(-prior*np.log(prior)-(1-prior)*np.log1p(-prior))
        result.append(dict(pre_current_state='DOWN' if s==-1 else 'UP',common_low=float(low),common_high=float(high),
            raw_F_expected_AUC=expected_auc(g,f,w),raw_F_signed_effect=expected_auc(g,f,w)-.5,
            fixed_nonlinear_oracle_expected_AUC=expected_auc(g,g,w),
            O1_expected_logloss=loss,O3_oracle_expected_logloss=loss,STATE_ONLY_expected_logloss=prior_loss,
            O3_increment_over_O1=0.,current_F_density_identical_support_but_different_shape=True))
    return dict(status='COMPLETE',known_truth='P(y=1|F,state)=sigmoid(2*(F^2-1)); no state interaction in true response.',
        generation_order='Pre-current state generated first; F conditional density proportional to exp(+/-1.2F) on[-2,2]; y depends only on F squared.',
        numerical_grid_points=2001,common_range_method='Population q10-q90 intersection from known precommitted densities',
        no_RP_or_hidden_data=True,no_fitted_candidates=True,not_a_sampling_p_value_or_independent_confirmation=True,records=result)
