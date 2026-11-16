"""Use the existing plotting environment, with the same two-worker queue."""
from pathlib import Path
import os
import sys
import subprocess
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import src.next8
from src.next8.common import BASE,read_json,digest


def main():
    lock=read_json(BASE/'FIGURE_RENDER_IMPLEMENTATION_LOCK_NEXT8.json')
    for name,sha in lock['source_sha256'].items():assert digest(ROOT/name)==sha,name
    executable=ROOT/'.figure-venv/Scripts/python.exe'
    if not executable.is_file():raise RuntimeError('Existing figure environment unavailable; no package installs authorized by this task')
    subprocess.run([str(executable),'-B',str(ROOT/'scripts/next8/render_next8_figures.py')],cwd=ROOT,check=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)


if __name__=='__main__':main()
