"""Use the already installed plotting environment, with no package installs."""
from pathlib import Path
import sys
import subprocess
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next9
from src.next9.common import BASE,read_json,digest


if __name__=='__main__':
    lock=read_json(BASE/'FIGURE_RENDERING_LOCK_NEXT9.json')
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    subprocess.run([str(ROOT/'.figure-venv/Scripts/python.exe'),'-X','utf8','-B',str(ROOT/'scripts/next9/plot_next9_results.py')],
        cwd=ROOT,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
