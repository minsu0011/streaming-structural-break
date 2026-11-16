"""Launch or explicitly resume the NEXT5 single-worker supervisor."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next5.scheduler import Supervisor


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true')
    Supervisor(resume=parser.parse_args().resume).run()
