"""Run the fixed actual D/T inference audit after prediction freeze."""
from pathlib import Path
import sys
import argparse
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.next8
from src.next8.common import BASE, read_json, digest, activity, note
from src.next8.forecast_causal_replay import audit_fold, summarize


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fold', type=int, choices=range(5))
    p.add_argument('--summarize', action='store_true')
    args = p.parse_args()
    if args.summarize == (args.fold is not None):
        p.error('Choose one fold or --summarize')
    lock = read_json(BASE / 'FORECAST_CAUSAL_REPLAY_IMPLEMENTATION_LOCK_NEXT8.json')
    for name, sha in lock['source_sha256'].items():
        assert digest(ROOT / name) == sha, name
    assert read_json(BASE / 'FORECAST_CAUSAL_REPLAY_TEST_GATE_NEXT8.json')['status'] == 'PASS'
    if args.summarize:
        with activity('Aggregate actual D/T future suffix and truncation replay', 'FORECAST_CAUSAL_REPLAY', 'B7_NORMALIZATION_AND_LINEAGE'):
            summarize()
    else:
        audit_fold(args.fold)
    note('Actual D/T causal replay complete', str(args.fold) if not args.summarize else 'Five-fold receipts retained in final payload', 'B7_NORMALIZATION_AND_LINEAGE')


if __name__ == '__main__':
    main()
