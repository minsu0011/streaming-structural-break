"""Descriptive simultaneous uncertainty for a fixed, already adaptive catalog."""
import numpy as np


def summarize_catalog(deltas,candidates,families):
    values=np.asarray(deltas,dtype=np.float64)
    if values.ndim!=2 or values.shape[0]<3 or values.shape[1]!=len(candidates) or len(families)!=len(candidates) or not np.isfinite(values).all():
        raise ValueError('Finite paired draw-by-candidate matrix required')
    if len(set(candidates))!=len(candidates):raise ValueError('Distinct candidate labels required')
    estimates=values[0];draws=values[1:];errors=draws-estimates
    radius=float(np.quantile(np.max(np.abs(errors),axis=1),.95))
    order=sorted(range(len(candidates)),key=lambda i:(-estimates[i],candidates[i]))
    winning=draws==draws.max(axis=1,keepdims=True);credit=winning/winning.sum(axis=1,keepdims=True)
    rows=[]
    for index in order:
        rows.append({'candidate':candidates[index],'family':families[index],'estimate_delta':float(estimates[index]),
            'pointwise_ci_lower':float(np.quantile(draws[:,index],.025)),'pointwise_ci_upper':float(np.quantile(draws[:,index],.975)),
            'conditional_catalog_envelope_lower':float(estimates[index]-radius),'conditional_catalog_envelope_upper':float(estimates[index]+radius),
            'fixed_OOF_bootstrap_winner_fraction':float(credit[:,index].mean())})
    family_rows=[]
    for family in sorted(set(families)):
        indices=[i for i,f in enumerate(families) if f==family];best=sorted(indices,key=lambda i:(-estimates[i],candidates[i]))[0]
        family_rows.append({'family':family,'completed_full_candidates':len(indices),'best_candidate':candidates[best],
            'best_estimate_delta':float(estimates[best]),'fraction_catalog_winner_from_family':float(credit[:,indices].sum(axis=1).mean())})
    first,second=order[:2];gap=draws[:,first]-draws[:,second]
    return {'candidates':rows,'families':family_rows,'catalog_envelope_radius':radius,'winner':candidates[first],
        'runner_up':candidates[second],'winner_gap':float(estimates[first]-estimates[second]),
        'fixed_pair_gap_ci_lower':float(np.quantile(gap,.025)),'fixed_pair_gap_ci_upper':float(np.quantile(gap,.975)),
        'fixed_pair_probability_gap_positive':float(np.mean(gap>0)),
        'scope':'Empirical centered-bootstrap envelope for this fixed OOF catalog. It does not correct prior adaptive feature engineering, parent selection, screening, historical DEV reuse or fitting uncertainty.',
        'winner_fraction_is_not_probability_of_true_best':True,'nominal_adaptive_familywise_coverage_claimed':False,'promotion_authority':False}
