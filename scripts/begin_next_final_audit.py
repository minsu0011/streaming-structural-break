"""Start the additional audit only after completed, interruption-adjusted research."""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.active_research_time import snapshot
from src.next.research_clock import ResearchClock, ResearchTimeError
from src.utils.artifacts import write_json, sha256, utc_now
from scripts.schedule_next_research import completed


def main():
    before = snapshot(ROOT)
    if not before['substantive_minimum_met']:
        raise ResearchTimeError('Interrupted/idle time does not count toward ten research hours')
    schedule = json.loads((ROOT/'configs/next_research_schedule.json').read_text(encoding='utf-8'))
    missing = [job['id'] for job in schedule['jobs'] if not completed(job)]
    if missing:
        raise ResearchTimeError('Required substantive research remains: '+', '.join(missing))
    out = ROOT/'artifacts/next/scheduler/SUBSTANTIVE_RESEARCH_DURATION.json'
    if before['final_audit_started'] or out.exists():
        raise ResearchTimeError('Preserve the first explicitly started final audit')
    ResearchClock(ROOT).begin_final_audit()
    result = snapshot(ROOT)
    write_json(out, {'status': 'PASS', 'created_utc': utc_now(), **result,
        'active_clock_source_sha256': sha256(ROOT/'src/next/active_research_time.py'),
        'start_driver_sha256': sha256(Path(__file__)), 'all_required_research_jobs_complete': True,
        'completed_schedule_jobs': len(schedule['jobs']), 'schedule_sha256': sha256(ROOT/'configs/next_research_schedule.json'),
        'scope': 'Original uninterrupted research plus resumed substantive work, excluding every conservatively audited interruption. Additional audit time starts only now and is not counted as research.'})
    print('ADDITIONAL AUDIT STARTED', result['substantive_research_seconds']/3600,
        'research hours; excluded', result['excluded_seconds']/3600, 'hours', flush=True)


if __name__ == '__main__':
    main()
