"""Conservative completed-work accounting, excluding idle and note leases."""
from src.next9.common import BASE, read_json, digest, union_seconds


def aggregate(records, admission_epoch=None):
    research = []
    audit = []
    excluded = {}
    seen = set()
    for r in records:
        if r['id'] in seen:
            raise ValueError('Duplicate activity ID')
        seen.add(r['id'])
        reason = None
        if r['status'] != 'COMPLETE':
            reason = 'NOT_COMPLETE'
        elif r['cpu_seconds'] <= .01:
            reason = 'INSUFFICIENT_PROCESS_CPU'
        elif r['kind'] in ('IDLE', 'WAIT', 'SLEEP', 'MONITOR', 'CONTROLLER_LEASE'):
            reason = 'NON_WORK_KIND'
        if reason:
            excluded[reason] = excluded.get(reason, 0) + 1
            continue
        a, b = r['started_epoch'], r['ended_epoch']
        if b < a:
            raise ValueError('Reversed work interval')
        if r['kind'] == 'FINAL_AUDIT':
            if admission_epoch is None or a < admission_epoch:
                raise ValueError('Final audit precedes research admission')
            audit.append((a, b))
        elif admission_epoch is None or b <= admission_epoch:
            research.append((a, b))
        elif a >= admission_epoch:
            excluded['POST_AUDIT_NON_AUDIT_HELPER'] = excluded.get('POST_AUDIT_NON_AUDIT_HELPER', 0) + 1
        else:
            raise ValueError('Unfinished research crosses final audit admission')
    return dict(productive_seconds=union_seconds(research), audit_seconds=union_seconds(audit),
                research_intervals=research, audit_intervals=audit, excluded_counts=excluded,
                successful_work_intervals=len(research) + len(audit),
                idle_or_note_leases_counted=False, parallel_or_nested_work_summed=False)


def snapshot(cache=None):
    cache = {} if cache is None else cache
    for path in (BASE / 'ledger').glob('*.json'):
        if path.stem not in cache:
            r = read_json(path)
            assert r['id'] == path.stem
            assert digest(path) == digest(BASE / 'runtime/jobs' / path.name)
            cache[path.stem] = r
    p = BASE / 'runtime/FINAL_AUDIT_ADMISSION_NEXT9.json'
    admission = read_json(p)['admitted_epoch'] if p.exists() else None
    return aggregate(cache.values(), admission)
