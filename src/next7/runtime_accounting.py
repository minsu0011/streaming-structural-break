"""Conservative sequential research/audit accounting gates."""
import math


def audit_admitted(admission, productive, session_sha256, minimum=36000.):
    return bool(admission and admission.get('status') == 'PASS'
        and admission.get('session_sha256') == session_sha256
        and admission.get('productive_seconds_observed', 0) >= minimum
        and productive >= minimum
        and math.isfinite(admission.get('admitted_epoch', float('nan'))))


def allocate_interval(start, stop, phase, *, admitted, admission_epoch=None):
    """Do not reclassify research jobs based on the latest controller note."""
    if not math.isfinite(start) or not math.isfinite(stop) or stop < start:
        raise ValueError('Finite ordered interval required')
    if phase == 'FINAL_AUDIT':
        if not admitted:
            return None
        start = max(start, admission_epoch)
        return ('audit_intervals', [start, stop]) if stop > start else None
    if admitted:
        stop = min(stop, admission_epoch)
    return ('intervals', [start, stop]) if stop > start else None
