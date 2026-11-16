"""Review package membership without producing or publishing a final ZIP."""
from pathlib import Path
import argparse
import re
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next5.artifacts import BASE, atomic_json, digest, utc_now, progress
from src.next5.handoff_payload import collect


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    assert re.fullmatch('[A-Za-z0-9_]+', args.tag)
    paths, files, excluded, missing = collect(require_final=False)
    out = BASE / 'handoff_inventory' / args.tag
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / 'RESULTS.json', {'created_utc': utc_now(), 'status': 'PREVIEW_NOT_FINAL_DELIVERY',
        'files': files, 'file_count': len(paths), 'bytes': sum(r['bytes'] for r in files.values()),
        'excluded': excluded, 'missing_final_reports': missing, 'new_archive_created': False,
        'raw_competition_data_included': False, 'real_point_OOF_included': False,
        'new_fits': 0, 'source_sha256': digest(Path(__file__)),
        'payload_policy_sha256': digest(ROOT / 'src/next5/handoff_payload.py')}, immutable=True)
    progress('Raw-free delivery inventory reviewed', str(len(paths)) + ' current source/evidence files; final ZIP remains gated')
    print('NEXT5 PAYLOAD PREVIEW', len(paths), 'files', sum(r['bytes'] for r in files.values()), 'bytes', len(missing), 'final reports pending', flush=True)


if __name__ == '__main__':
    main()
