"""Split an observed heartbeat at the real research-to-audit boundary."""
import math


def split_observed_interval(delta_seconds, *, observed, interval_end_wall, audit_started_wall=None):
    if type(observed) is not bool:
        raise ValueError('Observed activity must be an explicit boolean')
    if not math.isfinite(delta_seconds) or delta_seconds < 0 or not math.isfinite(interval_end_wall):
        raise ValueError('Finite nonnegative interval and finite end timestamp required')
    if audit_started_wall is not None and not math.isfinite(audit_started_wall):
        raise ValueError('Finite audit timestamp required')
    credit=delta_seconds if observed else 0.
    audit_overlap=0. if audit_started_wall is None else min(delta_seconds,max(0.,interval_end_wall-audit_started_wall))
    audit_credit=audit_overlap if observed else 0.
    return {'research_credit_seconds':credit-audit_credit,'audit_credit_seconds':audit_credit,
        'observed_credit_seconds':credit,'unobserved_seconds':delta_seconds-credit,
        'audit_wall_overlap_seconds':audit_overlap}
