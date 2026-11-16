"""Start tests with capacity for the legacy two-thread numerical oracle."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.next8

# This fresh test process alone needs capacity two. Its normal active count is
# one; the fixture restores one after the six legacy tests request two.
os.environ['NUMBA_NUM_THREADS'] = '2'
import numba
numba.set_num_threads(1)
import pytest

if __name__ == '__main__':
    raise SystemExit(pytest.main(sys.argv[1:]))
