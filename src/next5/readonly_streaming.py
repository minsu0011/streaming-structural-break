"""Read-only import adapter around the unchanged frozen NEXT5 scalar core."""
import sys
sys.dont_write_bytecode = True
from src.next.readonly_bootstrap import install
install()
from src.next5.streaming import infer, load_prepared
