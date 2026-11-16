"""Refresh completed evidence and follow the already fixed finalist ordering."""
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.next4.artifacts import BASE,read_json,utc_now


def main():
    tag='automatic_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
    subprocess.run([sys.executable,'scripts/review_next4_gate_evidence.py','--tag',tag],cwd=ROOT,check=True,creationflags=flags)
    subprocess.run([sys.executable,'scripts/nominate_next4_robust_finalist.py','--review-tag',tag],cwd=ROOT,check=True,creationflags=flags)


if __name__=='__main__':main()
