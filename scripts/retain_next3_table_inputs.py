"""Preserve mutable aggregate inputs before final tables replace their exports."""
from pathlib import Path
import shutil
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next3.artifacts import BASE, atomic_json, digest, progress, read_json, utc_now

MUTABLE = ('MODEL_RESULTS_NEXT3.csv', 'CV_RESULTS_NEXT3.csv', 'COUNTEREXAMPLE_RESULTS_NEXT3.csv',
           'LONG_NULL_RESULTS_NEXT3.csv', 'artifacts/next3/ledger/FIT_INVENTORY.json')


def main():
    prepared = BASE/'final_tables/prepared_v2'; record = read_json(prepared/'RESULTS.json')
    out = prepared/'input_snapshots'; out.mkdir(parents=True, exist_ok=False)
    mapping = {}
    for relative in MUTABLE:
        source = ROOT/relative; expected = record['source_evidence_sha256'][relative]
        assert digest(source) == expected
        target = out/relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target); assert digest(target) == expected
        mapping[relative] = {'snapshot': target.relative_to(ROOT).as_posix(), 'sha256': expected}
    for relative, sha in record['source_evidence_sha256'].items():
        assert digest(ROOT/relative) == sha
    atomic_json(prepared/'INPUT_RETENTION.json', {'status': 'PASS', 'created_utc': utc_now(),
        'prepared_receipt_sha256': digest(prepared/'RESULTS.json'), 'mutable_source_snapshots': mapping,
        'immutable_sources_checked': len(record['source_evidence_sha256'])-len(mapping),
        'purpose': 'Keep the exact prior aggregate exports when final publication changes the root tables. These snapshots contain no observation arrays or real point predictions.',
        'script_sha256': digest(Path(__file__))}, immutable=True)
    progress('Final table inputs retained', 'Five mutable aggregate exports preserved byte-for-byte; final publication can retain their original evidence hashes')
    print('NEXT3 FINAL TABLE INPUT SNAPSHOTS', len(mapping), 'PASS', flush=True)


if __name__ == '__main__': main()
