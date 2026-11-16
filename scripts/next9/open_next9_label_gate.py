from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,read_json,digest,activity
from src.next9.boundary import install_workspace_boundary
from src.next9.labels import open_label_gate

if __name__=='__main__':
    violations=install_workspace_boundary()
    lock=read_json(BASE/'ORIENTATION_SELECTION_IMPLEMENTATION_LOCK_NEXT9.json')
    for name,sha in lock['source_sha256'].items(): assert digest(ROOT/name)==sha,name
    with activity('Open scoped label projection only after all20 target-free candidate freezes','GUARD','B1_COMMON_CURRENT_RANGE'):
        open_label_gate()
        assert not violations
