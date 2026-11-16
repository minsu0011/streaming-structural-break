"""Research-only B2 entry; this is not the selected or submitted champion."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.next4.b2_inference import infer

__all__=['infer']
