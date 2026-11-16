"""Frozen CASE G and descriptive warnings, independent of model fitting."""


def assess(full_results,screen_results,bootstrap_results,policy):
    full={r['candidate']:r for r in full_results if r.get('status')=='COMPLETE' and r.get('scope')=='full'}
    failed_screens={r['candidate'] for r in screen_results if r.get('status')=='COMPLETE' and r.get('scope')=='screen' and r.get('screen_pass') is False}
    finished=set(full)|failed_screens
    best=max((r['mean_delta'] for r in full.values()),default=None)
    gate=policy['saturation'];case_g=len(finished)>=gate['structural_hypotheses_min'] and (best is None or best<gate['best_gain_strict_below'])
    ordered=sorted(full.values(),key=lambda r:r['mean_delta'],reverse=True)[:3]
    mixed_folds=sum(any(s-r>0 for s,r in zip(v['fold_scores'],v['reference_fold_scores'])) and
        any(s-r<0 for s,r in zip(v['fold_scores'],v['reference_fold_scores'])) for v in ordered)
    crossing=sum(r.get('status')=='PASS' and r['paired_all_time']['ci_2_5']<=0<=r['paired_all_time']['ci_97_5'] for r in bootstrap_results)
    warnings={'ten_completed_hypotheses_below_gain':case_g,'top_gains_have_mixed_fold_signs':mixed_folds>=2,
        'repeated_paired_intervals_cross_zero':crossing>=2}
    return {'completed_structural_hypotheses':len(finished),'completed_full_candidates':len(full),
        'screen_failed_candidates':len(failed_screens),'best_full_gain':best,'warning_conditions':warnings,
        'risk':'HIGH' if sum(warnings.values())>=2 or case_g else 'MONITOR',
        'stop_new_scientific_fits':case_g,'case_G':case_g,
        'counting':'A completed full result or a terminal failed screen counts once per preregistered structural candidate. Pending, merely registered and untested slots do not count.',
        'scope':'Conditional DEV warnings; unavailable ALT outcomes and subjective microvariant status are not asserted.'}
