"""Run fixed robustness and forensic diagnostics only for admitted late blends."""
from pathlib import Path
import json
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.late_evaluation import QUEUES, assert_late_time, validate_lock
from src.utils.artifacts import write_json, sha256, utc_now

if __name__ == '__main__':
    assert_late_time(ROOT)
    validate_lock(ROOT)
    out = ROOT/'artifacts/next/late_finalist_advancement'
    out.mkdir(parents=True, exist_ok=True)
    decisions = []
    for queue_path in QUEUES:
        queue = json.loads((ROOT/queue_path).read_text(encoding='utf-8'))
        for candidate in queue['candidates']:
            eid = candidate['experiment_id']
            path = ROOT/'artifacts/next/experiments'/eid/'RESULTS.json'
            report = json.loads(path.read_text(encoding='utf-8'))
            eligible = report['status'] == 'ROBUSTNESS_PENDING' and report.get('marginal_gate_pass', True)
            decision = {'candidate': eid, 'result_sha256': sha256(path),
                'primary_gate_status': report['status'], 'marginal_gate_pass': report.get('marginal_gate_pass'),
                'action': 'FIXED_STRESS_AND_FORENSIC' if eligible else 'NO_FINAL_ROLE_ADMISSION',
                'seal_rows': 0, 'reduced_new_uses': 0}
            write_json(out/(eid+'.json'), decision)
            if eligible:
                for stage in ('robustness', 'forensic'):
                    completed = ROOT/('artifacts/next/robustness/candidates' if stage == 'robustness' else 'artifacts/next/forensic')/eid/'RESULTS.json'
                    if not completed.exists():
                        subprocess.run([sys.executable, str(ROOT/'scripts'/f'{stage}_next_late.py'), '--candidate', eid], cwd=ROOT, check=True)
                    result = json.loads(completed.read_text(encoding='utf-8'))
                    if result['status'] != 'PASS':
                        raise RuntimeError('Late diagnostic did not pass')
                    decision[stage+'_result_sha256'] = sha256(completed)
            decisions.append(decision)
            print('LATE ADVANCEMENT', eid, decision['action'], flush=True)
    write_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(), 'decisions': decisions,
        'script_sha256': sha256(Path(__file__)),
        'implementation_lock_sha256': sha256(ROOT/'configs/next_late_implementation_lock.json'),
        'authority': 'No diagnostic can waive an already failed primary or prospective marginal-complexity gate.'})
