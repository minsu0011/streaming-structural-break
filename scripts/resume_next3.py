"""Resume preserved NEXT3 accounting and incomplete jobs, excluding offline gaps."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.next3.maintenance import Supervisor

if __name__ == '__main__':
    Supervisor(resume=True).run()
