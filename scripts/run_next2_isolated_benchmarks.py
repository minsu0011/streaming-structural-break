"""Sequential fresh processes; caller must wait for every other research worker to finish."""
from pathlib import Path
import sys
import json
import subprocess
import argparse
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.benchmark_next2_runtime import assert_isolated


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['core','components'],required=True);args=parser.parse_args()
    assert_isolated()
    if args.phase=='core':
        policy=json.loads((ROOT/'configs/next2/RUNTIME_MEASUREMENT_LOCK.json').read_text())
        commands=[['scripts/benchmark_next2_runtime.py','--variant',v] for v in policy['variant_order']]
    else:
        policy=json.loads((ROOT/'configs/next2/COMPONENT_MEASUREMENT_LOCK.json').read_text())
        commands=[['scripts/benchmark_next2_components.py','--candidate',r['experiment_id']] for r in policy['models']]
    for command in commands:
        subprocess.run([sys.executable,*command],cwd=ROOT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    print('NEXT2 ISOLATED PHASE COMPLETE',args.phase,flush=True)


if __name__=='__main__':
    main()
