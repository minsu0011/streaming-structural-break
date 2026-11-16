"""Read-only recheck of every authority file protected at NEXT7 entry."""
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.next7.common import BASE, read_json, save, digest, utc_now, activity, note


def main(label, final_audit=False):
    assert label and all(c.isalnum() or c in '_-' for c in label)
    name = f'audit/PROTECTED_AUTHORITY_{label}.json'
    assert not (BASE / name).exists()
    if final_audit:
        from scripts.next7.benchmark_runtime import audit_gate
        audit_gate()
    protected_path = BASE / 'PROTECTED_BEFORE_NEXT7_SHA256.json'
    protected = read_json(protected_path)
    mismatches = []
    count_bytes = 0
    with activity('Recheck all original protected authority files ' + label,
            'FINAL_AUDIT' if final_audit else 'SOURCE_AUTHORITY_FORENSIC', 'AUTONOMOUS_MAINTENANCE'):
        for index, (relative, expected) in enumerate(protected.items()):
            path = (ROOT / relative).resolve()
            assert path.is_relative_to(ROOT.resolve())
            if not path.is_file():
                mismatches.append({'path': relative, 'reason': 'MISSING'})
                continue
            actual = digest(path)
            count_bytes += path.stat().st_size
            if actual != expected:
                mismatches.append({'path': relative, 'expected_sha256': expected, 'actual_sha256': actual})
        save(name, {'status': 'PASS' if not mismatches else 'FAIL', 'created_utc': utc_now(),
            'protected_files': len(protected), 'bytes_rehashed': count_bytes, 'mismatches': mismatches,
            'protection_manifest_sha256': digest(protected_path), 'source_sha256': digest(__file__),
            'read_only': True, 'automatic_restoration_performed': False,
            'phase': 'FINAL_AUDIT' if final_audit else 'PRODUCTIVE_RESEARCH'})
        assert not mismatches, 'Protected source/artifact changed; see receipt, do not silently overwrite'
    note('Original authority remains unchanged', f'{len(protected)} protected files rehashed',
        'AUTONOMOUS_MAINTENANCE', phase='FINAL_AUDIT' if final_audit else 'PRODUCTIVE_RESEARCH')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    parser.add_argument('--final-audit', action='store_true')
    args = parser.parse_args()
    main(args.label, args.final_audit)
