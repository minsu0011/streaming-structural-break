"""Create a two-token-location derivative; never edit the frozen verifier."""
from pathlib import Path
import ast
import difflib
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next5.artifacts import BASE, atomic_json, digest, utc_now


def main():
    original = ROOT / 'scripts/verify_next5_streaming.py'
    destination = ROOT / 'scripts/verify_next5_streaming_readonly_core.py'
    before = original.read_text(encoding='utf-8')
    replacements = {
        'from src.next5.streaming import infer,load_prepared': 'from src.next5.readonly_streaming import infer,load_prepared',
        "out=BASE/'runtime_validation/api'/cid": "out=BASE/'runtime_validation/api_readonly_v1'/cid",
    }
    after = before
    for old, new in replacements.items():
        assert after.count(old) == 1
        after = after.replace(old, new)
    ast.parse(after)
    with destination.open('x', encoding='utf-8', newline='') as stream:
        stream.write(after)
    parent = BASE / 'maintenance/READONLY_IMPORT_REPAIR'
    diff = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
        fromfile=original.relative_to(ROOT).as_posix(), tofile=destination.relative_to(ROOT).as_posix()))
    (parent / 'VERIFIER_IMPORT_AND_OUTPUT_ONLY.diff').write_text(diff, encoding='utf-8')
    atomic_json(parent / 'DERIVATION.json', {'created_utc': utc_now(), 'status': 'PASS_EXACT_TWO_REPLACEMENTS',
        'original_sha256': digest(original), 'derived_sha256': digest(destination),
        'replacements': replacements, 'diff_sha256': digest(parent / 'VERIFIER_IMPORT_AND_OUTPUT_ONLY.diff'),
        'frozen_original_modified': False, 'numerical_validation_or_write_guards_changed': False,
        'fresh_output_directory_required': True, 'source_sha256': digest(Path(__file__))}, immutable=True)


if __name__ == '__main__':
    main()
