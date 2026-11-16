"""Build reviewable scalar tables; final publication is a separate gated step."""
from pathlib import Path
import argparse
import re
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next5.artifacts import BASE, atomic_json, read_json, digest, utc_now, progress
from src.next5.report_tables import ScalarEvidence, primary_tables, stability_tables, completed_stress_tables


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--complete', action='store_true')
    args = parser.parse_args()
    assert re.fullmatch('[A-Za-z0-9_]+', args.tag)
    if args.complete:
        session = read_json(BASE / 'SESSION.json')
        assert session.get('audit_started_utc'), 'Final tables require the separate final audit phase'
    evidence = ScalarEvidence()
    candidates = primary_tables(evidence)
    stability_tables(evidence, complete=args.complete)
    if args.complete:
        completed_stress_tables(evidence)
    else:
        evidence.pending.append('All 97 fitted synthetic artifacts; final exports deliberately deferred')
    out = BASE / 'report_tables' / args.tag
    out.mkdir(parents=True, exist_ok=False)
    files = evidence.write(out)
    atomic_json(out / 'RESULTS.json', {
        'created_utc': utc_now(), 'status': 'PASS_COMPLETE_SCALAR_TABLES' if args.complete else 'RESEARCH_PREVIEW',
        'candidate_count': len(candidates), 'full_candidate_count': sum(r['full_completed'] for r in candidates),
        'source_sha256': {name: digest(ROOT / name) for name in (
            'scripts/build_next5_scalar_tables.py', 'src/next5/report_tables.py')},
        'inputs_sha256': evidence.inputs, 'files': files, 'pending': evidence.pending,
        'new_fits': 0, 'new_random_draws': 0, 'real_point_predictions_read': False,
        'root_final_reports_published': False,
        'interpretation': 'PRIMARY, screen, matched ALT/H and historical references retain explicit populations; uncertainty is conditional and cannot override the frozen PRIMARY gate.',
    }, immutable=True)
    progress('Traceable scalar tables assembled', args.tag + '; original numeric precision and input hashes retained',
             phase='FINAL_AUDIT' if args.complete else 'RESEARCH')
    print('NEXT5 SCALAR TABLES', args.tag, len(files), 'files', len(evidence.pending), 'pending', flush=True)


if __name__ == '__main__':
    main()
