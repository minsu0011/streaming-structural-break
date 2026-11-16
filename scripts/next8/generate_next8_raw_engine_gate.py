"""Create only matched synthetic raw-law feature caches for the fixed control."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.next8
from src.next8.common import BASE, read_json, digest, activity
from src.next8.raw_engine_gate_control import generate_cache


def main():
    lock = read_json(BASE / 'RAW_ENGINE_GATE_IMPLEMENTATION_LOCK_NEXT8.json')
    for name, sha in lock['source_sha256'].items():
        assert digest(ROOT / name) == sha, name
    policy = read_json(BASE / 'RAW_ENGINE_GATE_CONTROL_POLICY_NEXT8.json')
    assert policy['banks'] == ['NULL', 'MEAN_PLUS2', 'SCALE_SD2']
    with activity('Generate matched raw-engine selection controls', 'SYNTHETIC_GATE_CONTROL', 'B7_NORMALIZATION_AND_LINEAGE'):
        generate_cache(BASE / 'raw_engine_gate_control/cache')


if __name__ == '__main__':
    main()
