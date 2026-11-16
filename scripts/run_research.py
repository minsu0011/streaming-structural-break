"""Public entry point for the wall-clock real-development research loop."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_real_research import admission,main

if __name__=="__main__":main()
