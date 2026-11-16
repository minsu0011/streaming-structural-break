"""Execute one registered task; stack dumps never masquerade as task progress."""
from pathlib import Path
import faulthandler
import os
import runpy
import sys
ROOT=Path(__file__).resolve().parents[1]


def main():
    args=sys.argv[1:]
    if args and args[0]=='--':args=args[1:]
    if not args:raise ValueError('Explicit worker argument list required')
    out=Path(os.environ['NEXT4_ATTEMPT_DIRECTORY']).resolve()
    if not out.is_relative_to(ROOT/'artifacts/next4/jobs'):raise ValueError('Owned NEXT4 attempt required')
    sys.path.insert(0,str(ROOT))
    with (out/'STACK_DIAGNOSTICS.log').open('x',encoding='utf-8') as diagnostics:
        faulthandler.enable(file=diagnostics)
        # The 1500-second asynchronous dump preceded a Windows native crash.
        # Preserve fatal-exception stacks; the supervisor independently observes
        # CPU/log progress and handles a truly idle owned process tree.
        # Evidence: artifacts/next4/maintenance/SYN_NATIVE_FAILURE_001/.
        try:
            if args[0]=='-m':
                if len(args)<2 or args[1]!='pytest':raise ValueError('Only pytest module allowed')
                sys.argv=['pytest',*args[2:]];runpy.run_module('pytest',run_name='__main__',alter_sys=True)
            else:
                path=(ROOT/args[0]).resolve()
                if not path.is_file() or not path.is_relative_to(ROOT/'scripts'):raise ValueError('Project script required')
                sys.argv=[str(path),*args[1:]];runpy.run_path(str(path),run_name='__main__')
        finally:faulthandler.cancel_dump_traceback_later()


if __name__=='__main__':main()
