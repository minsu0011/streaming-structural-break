"""Run the NEXT3 maintenance supervisor without a visible helper window."""
from pathlib import Path
import sys
import argparse
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next3.maintenance import Supervisor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    arguments = parser.parse_args()
    Supervisor(resume=arguments.resume).run()
