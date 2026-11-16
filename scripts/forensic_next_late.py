"""Reuse the frozen 24-slice forensic engine with an explicit late-OOF adapter.

Only its OOF loader is injected. Both adapter and unchanged engine hashes enter
the late implementation lock, and the injected loader adds them to provenance.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import forensic_next_candidate as engine
from src.next.late_evaluation import load_full_oof, validate_lock, assert_late_time

if __name__ == '__main__':
    assert_late_time(ROOT)
    validate_lock(ROOT)
    engine.load_full_oof = load_full_oof
    engine.main()
