"""Apply the already frozen full-CV admission rule to pending ablation families.

Every primary-gate-passing candidate receives the same 20 stress fits and 24
paired 2000-replicate fractures. Failed screens/full runs receive no waiver.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next.guard import SealGuard
from src.next.progress import checkpoint
from src.utils.artifacts import write_json, sha256, utc_now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queues', nargs='+', required=True)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    if not args.tag.replace('_', '').isalnum():
        raise ValueError('Invalid advancement tag')
    SealGuard(ROOT).check_state()
    out = ROOT/'artifacts/next/prospective_advancement'/args.tag
    out.mkdir(parents=True, exist_ok=True)
    queue_files = {relative: sha256(ROOT/relative) for relative in args.queues}
    scripts = ['scripts/robustness_next_candidate.py', 'scripts/forensic_next_candidate.py']
    identity = {'queues_sha256': queue_files, 'policy_sha256': sha256(ROOT/'ROBUSTNESS_SPLIT_LOCK.json'),
        'script_sha256': sha256(Path(__file__)), 'workers_sha256': {p: sha256(ROOT/p) for p in scripts},
        'admission': 'Every complete original-CV candidate whose existing frozen primary gates passed; never force a failed screen or full result.'}
    policy_path = out/'POLICY.json'
    if policy_path.exists():
        if json.loads(policy_path.read_text(encoding='utf-8')) != identity:
            raise RuntimeError('Advancement recipe changed')
    else:
        write_json(policy_path, identity)
    rows = []
    for relative in args.queues:
        queue = json.loads((ROOT/relative).read_text(encoding='utf-8'))
        for candidate in queue['candidates']:
            eid = candidate['experiment_id']
            path = ROOT/'artifacts/next/experiments'/eid/'RESULTS.json'
            original = json.loads(path.read_text(encoding='utf-8'))
            if original['candidate'] != candidate:
                raise RuntimeError('Candidate differs from frozen prospective queue')
            admitted = original['status'] in ('ROBUSTNESS_PENDING', 'NEW_DEV_CANDIDATE')
            row = {'candidate': eid, 'original_status': original['status'], 'original_sha256': sha256(path),
                'admitted_to_stress_and_forensic': admitted}
            if admitted:
                for script in scripts:
                    subprocess.run([sys.executable, str(ROOT/script), '--candidate', eid], cwd=ROOT, check=True)
                reports = [ROOT/f'artifacts/next/robustness/candidates/{eid}/RESULTS.json', ROOT/f'artifacts/next/forensic/{eid}/RESULTS.json']
                if any(json.loads(p.read_text(encoding='utf-8'))['status'] != 'PASS' for p in reports):
                    raise RuntimeError('Prospective finalist stress/forensic incomplete')
                row['completed_reports_sha256'] = {p.relative_to(ROOT).as_posix(): sha256(p) for p in reports}
            rows.append(row)
    write_json(out/'RESULTS.json', {'status': 'PASS', 'created_utc': utc_now(), 'identity': identity, 'candidates': rows,
        'admitted_count': sum(r['admitted_to_stress_and_forensic'] for r in rows), 'seal_rows': 0, 'reduced_new_uses': 0,
        'final_roles_assigned': False, 'original_screen_and_promotion_thresholds_unchanged': True})
    checkpoint(ROOT, 'prospective ablation and serial-evidence advancement', ['all newly admitted models receive unchanged robustness and forensic gates'],
        ['compare original-CV primary and backup eligibility', 'independent finalist reproduction'],
        artifacts=[(out/'RESULTS.json').relative_to(ROOT).as_posix()])
    print('PROSPECTIVE ADVANCEMENT COMPLETE', args.tag, sum(r['admitted_to_stress_and_forensic'] for r in rows), flush=True)


if __name__ == '__main__':
    main()
