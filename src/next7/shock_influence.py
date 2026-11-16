"""Finite differential influence of paired shocks, separate from lasting memory."""
import numpy as np

LOCAL_COLUMNS = (1,2,9,10,11,13,14,15,16,17,18,25,28,29,40,41,42,43,44)
CURRENT_STATE_COLUMNS = (0,7,12)
MEMORY_COLUMNS = (3,4,5,6,8,19,20,21,22,23,24,26,27,30,31,32,33,34,35,36,37,38,39)


def influence_bounds(changed_indices):
    changed=np.asarray(changed_indices)
    if changed.ndim!=1 or not len(changed) or changed.dtype.kind not in 'iu' or np.any(changed<0):
        raise ValueError('Nonempty nonnegative integer intervention positions required')
    last=int(changed.max())
    # The core has raw lag4; the S1 MAD block uses only its16 current raw values.
    affected_block=(last+4)//16
    # Four complete blocks, numbered affected_block+1 through+4, evict every
    # affected block from confirmation. Their final point is the common bound.
    fresh_equal_from=16*(affected_block+5)-1
    # The current-state automata have at most two further recovery-block steps.
    current_equal_from=fresh_equal_from+32
    return {'first_changed_point':int(changed.min()),'last_changed_point':last,
        'last_potentially_affected_core_block':affected_block,
        'local_evidence_equal_from_point':fresh_equal_from,
        'current_state_equal_from_point':current_equal_from,
        'no_finite_memory_equality_bound':True}


def compare_paths(intervened,control,changed_indices):
    a,b=np.asarray(intervened),np.asarray(control)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=45 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Two aligned finite G2 state paths required')
    bounds=influence_bounds(changed_indices)
    first=bounds['first_changed_point']
    fresh=bounds['local_evidence_equal_from_point']
    current=bounds['current_state_equal_from_point']
    assert current<len(a),'Complete follow-up past the analytic influence bound is required'
    assert np.array_equal(a[:first],b[:first])
    assert np.array_equal(a[fresh:,LOCAL_COLUMNS],b[fresh:,LOCAL_COLUMNS])
    assert np.array_equal(a[current:,CURRENT_STATE_COLUMNS],b[current:,CURRENT_STATE_COLUMNS])
    def max_error(cols,start):return float(np.max(np.abs(a[start:,cols]-b[start:,cols]),initial=0))
    return {**bounds,'prefix_bitwise_equal':True,'local_evidence_after_bound_bitwise_equal':True,
        'current_state_after_bound_bitwise_equal':True,
        'memory_max_difference_after_current_convergence':max_error(MEMORY_COLUMNS,current),
        'raw_ever_score_difference_at_end':float(a[-1,3]-b[-1,3]),
        'qualified_ever_score_difference_at_end':float(a[-1,30]-b[-1,30]),
        'raw_intervention_only_latch_at_end':bool(a[-1,3]>0 and b[-1,3]==0),
        'raw_control_only_latch_at_end':bool(a[-1,3]==0 and b[-1,3]>0),
        'qualified_intervention_only_latch_at_end':bool(a[-1,30]>0 and b[-1,30]==0),
        'qualified_control_only_latch_at_end':bool(a[-1,30]==0 and b[-1,30]>0),
        'qualified_episode_difference_at_end':float(a[-1,31]-b[-1,31]),
        'ever_memory_difference_not_equivalent_to_classifier_alarm':True}
